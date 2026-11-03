"""Score a completed, unlabeled batch with a frozen v3 study bundle."""

import argparse
import csv
import json
import re
import tempfile
from hashlib import sha256
from io import BytesIO, StringIO
from itertools import combinations
from numbers import Integral
from pathlib import Path
from pickle import UnpicklingError

import joblib
import numpy as np
import pandas as pd

from smartfab.models import predict_score
from smartfab.policies import blend_scores, capacity_mask, threshold_mask


def source_hashes():
    """Hash the same source inventory frozen by the study."""
    root = Path(__file__).resolve().parents[1]
    paths = [*sorted((root / "smartfab").glob("*.py")),
             *sorted((root / "secom").glob("*.py")), root / "pyproject.toml", root / "uv.lock"]
    return {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}


def predict_batch(bundle, frame):
    """Return unit_id, uncalibrated score, capacity review and frozen threshold alert."""
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 3:
        raise ValueError("A schema_version 3 bundle is required.")
    columns, members = bundle["sensor_columns"], bundle["members"]
    if (not isinstance(columns, list) or not columns
            or any(not isinstance(c, str) or not c or c == "unit_id" for c in columns)
            or len(set(columns)) != len(columns)):
        raise ValueError("Bundle sensor_columns must be unique sensor names.")
    blend = bundle["is_rank_blend"]
    if (not isinstance(blend, bool) or not isinstance(members, list)
            or len(members) != (2 if blend else 1)
            or bundle["winner"] != ("rank_blend_top2" if blend else members[0]["spec"]["id"])
            or bundle["review_fraction"] != 0.20):
        raise ValueError("Bundle winner, members or 20% review budget is inconsistent.")
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise ValueError("Input must be a nonempty DataFrame with unique sensor headers.")
    if set(frame.columns) - {"unit_id"} != set(columns):
        raise ValueError("Input sensor headers must match training exactly; labels are invalid.")
    ids = list(range(len(frame)))
    if "unit_id" in frame:
        ids = frame["unit_id"].tolist()
        if any(not ((isinstance(v, Integral) and not isinstance(v, (bool, np.bool_)))
                    or (isinstance(v, str) and re.fullmatch(r"[+-]?[0-9]+", v))) for v in ids):
            raise ValueError("unit_id must contain unique integers.")
        ids = [int(v) for v in ids]
        if len(set(ids)) != len(ids):
            raise ValueError("unit_id must contain unique integers.")
    ordered = frame.loc[:, columns]
    numeric = ordered.mask(ordered.eq("")).apply(pd.to_numeric, errors="raise")
    if any(np.iscomplexobj(numeric[c]) for c in columns):
        raise ValueError("Sensor values must be finite real numbers or blank.")
    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError("Sensor values must be finite real numbers or blank.")
    scores = []
    for member in members:
        pipeline = member["pipeline"]
        pruner = pipeline.named_steps["sensors"]
        mask = np.asarray(pruner.keep_mask_)
        if (mask.dtype != bool or mask.shape != (len(columns),) or not mask.any()
                or not np.array_equal(pruner.feature_names_in_, columns)):
            raise ValueError("Fitted sensor mask/schema differs from bundle sensor_columns.")
        if np.isnan(values[:, mask]).all(axis=1).any():
            raise ValueError("Every row needs an observed retained sensor for every member.")
        scores.append(predict_score(member["spec"], pipeline, numeric))
    score = blend_scores(scores) if blend else scores[0]
    return pd.DataFrame({
        "unit_id": ids, "score": score,
        "review": capacity_mask(score, ids, bundle["review_fraction"], bundle["seed"]).astype(int),
        "threshold_alert": threshold_mask(score, bundle["secondary_threshold"]).astype(int),
    })


def verify_recipe(bundle, protocol_bytes, selection_bytes):
    """Verify actual recipe bytes, frozen decisions and the complete source inventory."""
    for name, content in (("protocol", protocol_bytes), ("selection", selection_bytes)):
        if sha256(content).hexdigest() != bundle[f"{name}_sha256"]:
            raise ValueError(f"{name} file SHA256 differs from bundle.")
    protocol, selection = json.loads(protocol_bytes), json.loads(selection_bytes)
    specs = [m["spec"] for m in bundle["members"]]
    if (bundle.get("schema_version") != 3 or protocol.get("schema_version") != 3
            or selection["protocol_sha256"] != bundle["protocol_sha256"]
            or selection["winner"] != bundle["winner"]
            or selection["members"] != [s["id"] for s in specs]
            or selection["member_specs"] != specs
            or selection["is_rank_blend"] != bundle["is_rank_blend"]
            or selection["threshold_policy"]["threshold"] != bundle["secondary_threshold"]
            or selection["primary_review_fraction"] != bundle["review_fraction"]
            or protocol["config"]["review_fraction"] != bundle["review_fraction"]
            or protocol["config"]["seed"] != bundle["seed"]
            or any(s not in protocol["catalog"] for s in specs)):
        raise ValueError("Frozen protocol, selection and bundle recipe are inconsistent.")
    actual = source_hashes()
    if actual != protocol["source_sha256"]:
        raise ValueError("Frozen source SHA256 differs; use the original study source.")
    return actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "protocol", "selection", "input", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    metadata_path = args.output.with_suffix(".metadata.json")
    try:
        paths = [args.model, args.protocol, args.selection, args.input, args.output, metadata_path]
        for left, right in combinations(paths, 2):
            if left.resolve() == right.resolve() or (
                left.exists() and right.exists() and left.samefile(right)
            ):
                raise ValueError("Model, protocol, selection, input, output and metadata must be distinct.")
        for path in (args.output, metadata_path):
            if path.is_dir():
                raise ValueError(f"Output is a directory: {path}")
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"{path} exists; use --overwrite to replace predictions.")
        content = {name: getattr(args, name).read_bytes()
                   for name in ("model", "protocol", "selection", "input")}
        # Load only trusted joblib models: deserialization executes Python code.
        bundle = joblib.load(BytesIO(content["model"]))
        actual_source = verify_recipe(bundle, content["protocol"], content["selection"])
        if b"\x00" in content["input"]:
            raise ValueError("Input CSV cannot contain NUL bytes.")
        reader = csv.reader(StringIO(content["input"].decode("utf-8-sig")), strict=True)
        header = next(reader, [])
        if not header or len(set(header)) != len(header):
            raise ValueError("CSV needs unique sensor headers; duplicate headers are invalid.")
        rows = list(reader)
        if any(len(row) != len(header) for row in rows):
            raise ValueError("Every CSV row must match the header width.")
        frame = pd.DataFrame(rows, columns=header)
        predictions = predict_batch(bundle, frame)
        id_source = "unit_id" if "unit_id" in frame else "batch_position"
        id_note = ("Stable unit_id values preserve score ties across row order changes."
                   if id_source == "unit_id" else
                   "unit_id absent: zero-based batch positions used; stable IDs are needed "
                   "to preserve score ties when row order changes.")
        metadata = {
            "schema_version": 3,
            **{f"{name}_sha256": sha256(value).hexdigest() for name, value in content.items()},
            "source_sha256": actual_source,
            "prediction_code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "rows": len(predictions), "winner": bundle["winner"],
            "members": [m["spec"]["id"] for m in bundle["members"]],
            "review_fraction": bundle["review_fraction"],
            "review_budget": int(predictions.review.sum()), "seed": bundle["seed"],
            "secondary_threshold": bundle["secondary_threshold"],
            "unit_id_source": id_source, "unit_id_note": id_note,
            "score_meaning": "Uncalibrated model scores; larger values indicate failure.",
            "review_meaning": "Exact floor(20% * rows) review capacity for this completed batch; SHA256 ID ties.",
            "alert_meaning": "threshold_alert uses the frozen secondary threshold; None means zero alerts. No capacity guarantee.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        staged = []
        try:
            # Close both staged files before publishing either destination.
            for destination in (args.output, metadata_path):
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", newline="", dir=destination.parent,
                    prefix=f".{destination.name}.", suffix=".tmp", delete=False,
                ) as temporary:
                    staged.append((Path(temporary.name), destination))
                    if destination == args.output:
                        predictions.to_csv(temporary, index=False)
                    else:
                        temporary.write(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
            for temporary_path, destination in staged:
                if args.overwrite:
                    temporary_path.replace(destination)
                else:
                    destination.hardlink_to(temporary_path)
        finally:
            for temporary_path, _ in staged:
                temporary_path.unlink(missing_ok=True)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, EOFError,
            UnpicklingError, csv.Error) as error:
        parser.error(str(error))
    print(f"Wrote {len(predictions)} predictions to {args.output} and {metadata_path}. {id_note}")


if __name__ == "__main__":
    main()
