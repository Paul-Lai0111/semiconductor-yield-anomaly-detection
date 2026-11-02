"""Batch inference using a trusted study joblib bundle; joblib loading executes Python.

Scores are uncalibrated failure-class model outputs, not validated factory risks.
Inputs need only the original sensors; blank cells are missing, other text is invalid.
"""

import argparse
import csv
import json
import tempfile
from hashlib import sha256
from io import BytesIO, StringIO
from itertools import combinations
from numbers import Real
from pathlib import Path
from pickle import UnpicklingError

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from secom.modeling import CANDIDATES, make_pipeline


def predict_batch(bundle, frame):
    """Score unlabeled rows without fitting; row_id is the zero-based batch position.

    Sensor headers may arrive in any order. Numeric strings and missing cells are
    accepted; every row must observe at least one sensor kept during training.
    """
    required = {"candidate", "threshold", "sensor_columns", "pipeline"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("Bundle needs candidate, threshold, sensor_columns and pipeline.")
    candidate, threshold = bundle["candidate"], bundle["threshold"]
    if not isinstance(candidate, str) or candidate not in CANDIDATES:
        raise ValueError("Bundle candidate is unsupported.")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not np.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Bundle threshold must be finite and in [0, 1].")
    columns, pipeline = bundle["sensor_columns"], bundle["pipeline"]
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(name, str) or not name for name in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ValueError("Bundle sensor_columns must be unique, nonempty sensor names.")
    expected = make_pipeline(candidate, seed=0, trees=1, jobs=1)
    if not callable(getattr(pipeline, "predict_proba", None)) or [
        (name, type(step)) for name, step in getattr(pipeline, "steps", [])
    ] != [(name, type(step)) for name, step in expected.steps]:
        raise ValueError("Bundle pipeline does not support the declared candidate.")
    for step in pipeline.named_steps.values():
        check_is_fitted(step)
    if not np.array_equal(pipeline.classes_, [0, 1]):
        raise ValueError("Bundle pipeline must predict classes [0, 1] in that order.")
    if getattr(pipeline, "n_features_in_", len(columns)) != len(columns) or not np.array_equal(
        getattr(pipeline, "feature_names_in_", columns), columns
    ):
        raise ValueError("Bundle sensor schema differs from the fitted pipeline.")
    sensors = pipeline.named_steps.get("sensors")
    mask = np.ones(len(columns), dtype=bool) if sensors is None else sensors.keep_mask_
    mask = np.asarray(mask)
    if mask.shape != (len(columns),) or mask.dtype != bool or not mask.any():
        raise ValueError("Fitted sensor mask is invalid.")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Input must be a nonempty sensor DataFrame.")
    if not frame.columns.is_unique or set(frame.columns) != set(columns):
        raise ValueError("Input sensor headers must match training exactly, without duplicates.")
    ordered = frame.loc[:, columns]
    numeric = ordered.mask(ordered.eq("")).apply(pd.to_numeric, errors="raise")
    if any(np.iscomplexobj(numeric[name]) for name in columns):
        raise ValueError("Sensor values must be real numbers or missing.")
    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError("Sensor values cannot contain infinity.")
    if np.isnan(values[:, mask]).all(axis=1).any():
        raise ValueError(
            "Every row needs an observed retained sensor; all-missing rows are invalid."
        )
    probabilities = np.asarray(pipeline.predict_proba(numeric), dtype=float)
    if (
        probabilities.shape != (len(frame), 2)
        or not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Pipeline must return finite two-class scores in [0, 1] for every row.")
    scores = probabilities[:, 1]
    return pd.DataFrame(
        {
            "row_id": np.arange(len(frame)),
            "score": scores,
            "alert": (scores >= threshold).astype(int),
        }
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("results/protocol.json"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing prediction files."
    )
    args = parser.parse_args(argv)
    metadata_path = args.output.with_suffix(".metadata.json")
    try:
        paths = [args.model, args.protocol, args.input, args.output, metadata_path]
        for left, right in combinations(paths, 2):
            if left.resolve() == right.resolve() or (
                left.exists() and right.exists() and left.samefile(right)
            ):
                raise ValueError(
                    "Model, protocol, input, output and metadata paths must be distinct."
                )
        for path in (args.output, metadata_path):
            if path.is_dir():
                raise ValueError(f"Output is a directory: {path}")
            if path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"{path} exists; use --overwrite to replace prediction files."
                )
        model_bytes = args.model.read_bytes()
        protocol_bytes = args.protocol.read_bytes()
        input_bytes = args.input.read_bytes()
        if b"\x00" in input_bytes:
            raise ValueError("Input CSV cannot contain NUL bytes.")
        bundle = joblib.load(BytesIO(model_bytes))
        protocol_hash = sha256(protocol_bytes).hexdigest()
        if not isinstance(bundle, dict) or bundle.get("protocol_sha256") != protocol_hash:
            raise ValueError("Protocol file SHA256 differs from bundle protocol_sha256.")
        protocol = json.loads(protocol_bytes)
        if (
            not isinstance(protocol, dict)
            or protocol.get("schema_version") != 1
            or not isinstance(bundle.get("candidate"), str)
            or protocol.get("selected_candidate") != bundle["candidate"]
            or not isinstance(protocol.get("thresholds"), dict)
            or protocol["thresholds"].get(bundle["candidate"]) != bundle.get("threshold")
        ):
            raise ValueError(
                "Protocol schema, selected candidate or frozen threshold is inconsistent."
            )
        root = Path(__file__).resolve().parents[1]
        source_hash = sha256()
        for path in [
            root / "main.py",
            root / "pyproject.toml",
            root / "uv.lock",
            *sorted((root / "secom").glob("*.py")),
        ]:
            source_hash.update(path.relative_to(root).as_posix().encode())
            source_hash.update(path.read_bytes())
        if source_hash.hexdigest() != protocol.get("source_sha256"):
            raise ValueError("Frozen source SHA256 differs; use the model's original study source.")
        reader = csv.reader(StringIO(input_bytes.decode("utf-8-sig")), strict=True)
        header = next(reader, [])
        if not header or len(set(header)) != len(header):
            raise ValueError("CSV needs a unique sensor header; duplicate headers are invalid.")
        if any(len(row) != len(header) for row in reader):
            raise ValueError("Every CSV row must match the header width, without blank records.")
        frame = pd.read_csv(
            BytesIO(input_bytes),
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
            skip_blank_lines=False,
        )
        predictions = predict_batch(bundle, frame)
        metadata = {
            "bundle_sha256": sha256(model_bytes).hexdigest(),
            "input_sha256": sha256(input_bytes).hexdigest(),
            "protocol_sha256": protocol_hash,
            "source_sha256": source_hash.hexdigest(),
            "prediction_code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "candidate": bundle["candidate"],
            "threshold": float(bundle["threshold"]),
            "rows": len(predictions),
            "score_meaning": (
                "Uncalibrated failure-class (1) model score; alert = score >= frozen threshold. "
                "Research model, not validated for factory deployment."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        staged = []
        try:
            # Finish and close both files before publishing either destination.
            for destination in (args.output, metadata_path):
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="",
                    dir=destination.parent,
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    staged.append((Path(temporary.name), destination))
                    if destination == args.output:
                        predictions.to_csv(temporary, index=False)
                    else:
                        temporary.write(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
                    temporary.flush()
            for temporary_path, destination in staged:
                if args.overwrite:
                    temporary_path.replace(destination)
                else:
                    destination.hardlink_to(temporary_path)
        finally:
            for temporary_path, _ in staged:
                temporary_path.unlink(missing_ok=True)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        EOFError,
        UnpicklingError,
        csv.Error,
    ) as error:
        parser.error(str(error))
    print(f"Wrote {len(predictions)} predictions to {args.output} and {metadata_path}.")


if __name__ == "__main__":
    main()
