"""Unlabeled inference and isolated CSV/provenance regressions with tiny real models."""

import json
import os
import subprocess
import sys
import tempfile
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts import predict
from secom.modeling import CANDIDATES, fit_pipeline, make_pipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def training():
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(rng.normal(size=(24, 4)), columns=[f"sensor_{i:03d}" for i in range(4)])
    frame.loc[::3, "sensor_001"] = np.nan
    frame["sensor_002"] = 7.0
    frame["sensor_003"] = np.nan
    return frame, np.tile([0, 1], 12)


@pytest.fixture
def bundle(training):
    frame, labels = training
    return {
        "candidate": "rf_knn",
        "pipeline": fit_pipeline("rf_knn", frame, labels, seed=17, trees=2, jobs=1),
        "sensor_columns": frame.columns.tolist(),
        "threshold": 0.5,
    }


def test_scores_match_direct_model_without_fitting_or_mutating(bundle, training, monkeypatch):
    frame = training[0].iloc[:5].copy()
    original = frame.copy(deep=True)
    pipeline = bundle["pipeline"]
    direct = pipeline.predict_proba(frame)[:, 1]
    bundle["threshold"] = float(direct[0])
    mask = pipeline["sensors"].keep_mask_.copy()
    mean = pipeline["scale"].mean_.copy()
    donors = pipeline["impute"]._fit_X.copy()

    def forbid_fit(*args, **kwargs):
        raise AssertionError("Inference must never fit anything.")

    for estimator in (pipeline, *pipeline.named_steps.values()):
        monkeypatch.setattr(estimator, "fit", forbid_fit)
    result = predict.predict_batch(bundle, frame.iloc[:, ::-1])

    assert result.columns.tolist() == ["row_id", "score", "alert"]
    np.testing.assert_array_equal(result.row_id, np.arange(len(frame)))
    np.testing.assert_array_equal(result.score, direct)
    np.testing.assert_array_equal(result.alert, (direct >= bundle["threshold"]).astype(int))
    assert result.alert.iloc[0] == 1
    assert pd.api.types.is_integer_dtype(result.alert)
    np.testing.assert_array_equal(pipeline["sensors"].keep_mask_, mask)
    np.testing.assert_array_equal(pipeline["scale"].mean_, mean)
    np.testing.assert_array_equal(pipeline["impute"]._fit_X, donors)
    pd.testing.assert_frame_equal(frame, original)


def test_numeric_strings_blank_cells_and_positional_row_ids(bundle, training):
    frame = training[0].iloc[:3].copy()
    expected = predict.predict_batch(bundle, frame)
    strings = frame.astype(str).mask(frame.isna(), "")
    strings.index = [90, 90, -1]
    pd.testing.assert_frame_equal(predict.predict_batch(bundle, strings), expected)


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_all_current_candidate_pipelines_are_supported(candidate, training):
    frame, labels = training
    bundle = {
        "candidate": candidate,
        "pipeline": fit_pipeline(candidate, frame, labels, seed=17, trees=2, jobs=1),
        "sensor_columns": frame.columns.tolist(),
        "threshold": 0.5,
    }
    result = predict.predict_batch(bundle, frame.iloc[:3])
    np.testing.assert_array_equal(
        result.score, bundle["pipeline"].predict_proba(frame.iloc[:3])[:, 1]
    )
    with pytest.raises(ValueError, match="observed retained sensor"):
        predict.predict_batch(bundle, frame.iloc[:1] * np.nan)


@pytest.mark.parametrize(
    "change", ["missing", "extra", "duplicate", "empty", "all_raw_missing", "all_retained_missing"]
)
def test_rejects_invalid_batches(bundle, training, change):
    frame = training[0].iloc[:3].copy()
    if change == "missing":
        frame = frame.drop(columns="sensor_003")
    elif change == "extra":
        frame["label"] = 0
    elif change == "duplicate":
        frame.columns = ["sensor_000", "sensor_000", "sensor_002", "sensor_003"]
    elif change == "empty":
        frame = frame.iloc[:0]
    elif change == "all_raw_missing":
        frame.iloc[1] = np.nan
    else:
        frame.loc[frame.index[1], ["sensor_000", "sensor_001"]] = np.nan
        assert frame.loc[frame.index[1], "sensor_002"] == 7
    with pytest.raises(ValueError):
        predict.predict_batch(bundle, frame)


@pytest.mark.parametrize(
    "value", ["garbage", "NaN", "null", "NA", "1e999", np.inf, -np.inf, 1 + 2j]
)
def test_rejects_non_numeric_or_infinite_sensors(bundle, training, value):
    frame = training[0].iloc[:3].astype(object)
    frame.iloc[1, 0] = value
    with pytest.raises((ValueError, TypeError)):
        predict.predict_batch(bundle, frame)


@pytest.mark.parametrize("threshold", [-0.01, 1.01, np.nan, np.inf, -np.inf, "0.5", None, True])
def test_rejects_invalid_thresholds(bundle, training, threshold):
    bundle["threshold"] = threshold
    with pytest.raises(ValueError, match="threshold"):
        predict.predict_batch(bundle, training[0])


@pytest.mark.parametrize("threshold", [0, 1])
def test_threshold_endpoints_are_valid(bundle, training, threshold):
    bundle["threshold"] = threshold
    result = predict.predict_batch(bundle, training[0])
    np.testing.assert_array_equal(result.alert, (result.score >= threshold).astype(int))


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "candidate",
        "schema_duplicate",
        "schema_empty",
        "schema_order",
        "schema_width",
        "pipeline",
        "unfitted",
        "mislabeled",
        "classes",
        "mask_shape",
        "mask_empty",
        "mask_dtype",
    ],
)
def test_rejects_invalid_bundles(bundle, training, change):
    if change == "missing":
        del bundle["threshold"]
    elif change == "candidate":
        bundle["candidate"] = "unsupported"
    elif change == "schema_duplicate":
        bundle["sensor_columns"][1] = bundle["sensor_columns"][0]
    elif change == "schema_empty":
        bundle["sensor_columns"] = []
    elif change == "schema_order":
        bundle["sensor_columns"] = bundle["sensor_columns"][::-1]
    elif change == "schema_width":
        bundle["sensor_columns"] = bundle["sensor_columns"][:-1]
    elif change == "pipeline":
        bundle["pipeline"] = object()
    elif change == "unfitted":
        bundle["pipeline"] = make_pipeline("rf_knn", seed=0, trees=2, jobs=1)
    elif change == "mislabeled":
        bundle["candidate"] = "rf_median"
    elif change == "classes":
        bundle["pipeline"]["model"].classes_ = np.array([1, 0])
    elif change == "mask_shape":
        bundle["pipeline"]["sensors"].keep_mask_ = np.ones(2, dtype=bool)
    elif change == "mask_empty":
        bundle["pipeline"]["sensors"].keep_mask_[:] = False
    else:
        bundle["pipeline"]["sensors"].keep_mask_ = np.ones(4, dtype=int)
    with pytest.raises(ValueError):
        predict.predict_batch(bundle, training[0])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -0.1, 1.1, "shape"])
def test_rejects_invalid_model_scores(bundle, training, monkeypatch, bad):
    probabilities = np.full((len(training[0]), 2), 0.5)
    if bad == "shape":
        probabilities = probabilities[:, 1]
    else:
        probabilities[0, 1] = bad
    monkeypatch.setattr(bundle["pipeline"], "predict_proba", lambda _: probabilities)
    with pytest.raises(ValueError, match="finite two-class scores"):
        predict.predict_batch(bundle, training[0])


@pytest.fixture
def files(tmp_path, bundle, training):
    source_hash = sha256()
    for path in [
        ROOT / "main.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        *sorted((ROOT / "secom").glob("*.py")),
    ]:
        source_hash.update(path.relative_to(ROOT).as_posix().encode())
        source_hash.update(path.read_bytes())
    protocol = {
        "schema_version": 1,
        "source_sha256": source_hash.hexdigest(),
        "selected_candidate": bundle["candidate"],
        "thresholds": {bundle["candidate"]: bundle["threshold"]},
    }
    paths = {
        "model": tmp_path / "selected_model.joblib",
        "protocol": tmp_path / "protocol.json",
        "input": tmp_path / "measurements.csv",
        "output": tmp_path / "nested" / "predictions.csv",
        "metadata": tmp_path / "nested" / "predictions.metadata.json",
    }
    paths["protocol"].write_text(json.dumps(protocol), encoding="utf-8")
    bundle["protocol_sha256"] = sha256(paths["protocol"].read_bytes()).hexdigest()
    joblib.dump(bundle, paths["model"], compress=3)
    training[0].iloc[:4, ::-1].to_csv(paths["input"], index=False)
    return paths


def arguments(files):
    return [
        value
        for name in ("model", "protocol", "input", "output")
        for value in (f"--{name}", str(files[name]))
    ]


def test_module_cli_writes_exact_scores_and_actual_file_hashes(files, bundle, training):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "scripts.predict", *arguments(files)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "Wrote 4 predictions" in result.stdout
    actual = pd.read_csv(files["output"])
    expected = predict.predict_batch(bundle, training[0].iloc[:4])
    pd.testing.assert_frame_equal(actual, expected)
    metadata = json.loads(files["metadata"].read_text())
    for field, name in [
        ("bundle_sha256", "model"),
        ("input_sha256", "input"),
        ("protocol_sha256", "protocol"),
    ]:
        assert metadata[field] == sha256(files[name].read_bytes()).hexdigest()
    assert (
        metadata["prediction_code_sha256"]
        == sha256(Path(predict.__file__).read_bytes()).hexdigest()
    )
    assert metadata["candidate"] == bundle["candidate"]
    assert metadata["threshold"] == bundle["threshold"]
    assert metadata["rows"] == 4
    assert "Uncalibrated" in metadata["score_meaning"]
    assert "not validated for factory" in metadata["score_meaning"]


@pytest.mark.parametrize("existing", ["output", "metadata", "both"])
def test_cli_preserves_existing_files_unless_overwrite_is_explicit(files, existing):
    files["output"].parent.mkdir()
    names = ["output", "metadata"] if existing == "both" else [existing]
    for name in names:
        files[name].write_bytes(b"do not replace\n")
    with pytest.raises(SystemExit) as error:
        predict.main(arguments(files))
    assert error.value.code == 2
    for name in names:
        assert files[name].read_bytes() == b"do not replace\n"
    for name in {"output", "metadata"} - set(names):
        assert not files[name].exists()
    predict.main([*arguments(files), "--overwrite"])
    assert len(pd.read_csv(files["output"])) == 4
    assert json.loads(files["metadata"].read_text())["rows"] == 4


@pytest.mark.parametrize("failure", ["second_open", "sidecar_write"])
def test_staging_failure_preserves_both_existing_outputs(files, monkeypatch, capsys, failure):
    files["output"].parent.mkdir()
    original = {"output": b"old predictions\n", "metadata": b"old metadata\n"}
    for name, content in original.items():
        files[name].write_bytes(content)
    real_temporary_file = tempfile.NamedTemporaryFile
    calls = 0

    def failing_temporary_file(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2 and failure == "second_open":
            raise PermissionError("injected sidecar staging failure")
        staged = real_temporary_file(*args, **kwargs)
        if calls == 2:

            def fail_write(text):
                staged.file.write(text[:8])
                raise PermissionError("injected sidecar staging failure")

            monkeypatch.setattr(staged, "write", fail_write)
        return staged

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", failing_temporary_file)
    with pytest.raises(SystemExit) as error:
        predict.main([*arguments(files), "--overwrite"])
    assert error.value.code == 2
    assert "injected sidecar staging failure" in capsys.readouterr().err
    for name, content in original.items():
        assert files[name].read_bytes() == content
    assert set(files["output"].parent.iterdir()) == {files["output"], files["metadata"]}


@pytest.mark.parametrize(
    "source,destination",
    [
        ("output", "input"),
        ("output", "model"),
        ("output", "protocol"),
        ("input", "model"),
        ("input", "protocol"),
        ("model", "protocol"),
        ("input", "metadata"),
        ("model", "metadata"),
        ("protocol", "metadata"),
    ],
)
def test_cli_rejects_path_collisions_even_with_overwrite(files, source, destination, capsys):
    original = {name: files[name].read_bytes() for name in ("input", "model", "protocol")}
    paths = {**files, source: files[destination]}
    with pytest.raises(SystemExit):
        predict.main([*arguments(paths), "--overwrite"])
    assert "must be distinct" in capsys.readouterr().err
    for name, content in original.items():
        assert files[name].read_bytes() == content


@pytest.mark.parametrize("link", ["symlink", "hardlink"])
def test_cli_rejects_output_aliases_of_inputs(files, link, capsys):
    files["output"].parent.mkdir()
    original = files["input"].read_bytes()
    if link == "symlink":
        files["output"].symlink_to(files["input"])
    else:
        files["output"].hardlink_to(files["input"])
    with pytest.raises(SystemExit):
        predict.main([*arguments(files), "--overwrite"])
    assert "must be distinct" in capsys.readouterr().err
    assert files["input"].read_bytes() == original


@pytest.mark.parametrize(
    "content",
    [
        '"sensor_000",sensor_000,sensor_002,sensor_003\n1,2,7,\n',
        "sensor_000,sensor_001,sensor_002,sensor_003\n1,2,7,,9\n",
        "sensor_000,sensor_001,sensor_002,sensor_003\n1,2,7\n",
        "sensor_000,sensor_001,sensor_002,sensor_003\n1,2,7,\n\n",
        "sensor_000,sensor_001,sensor_002,sensor_003\n",
        "sensor_000,sensor_001,sensor_002,sensor_003\n,,,\n",
        "sensor_000,sensor_001,sensor_002,sensor_003\n,,7,\n",
        "sensor_000,sensor_001,sensor_002,sensor_003\nNaN,2,7,\n",
        "sensor_000,sensor_001,sensor_002,sensor_003\ninf,2,7,\n",
        "",
    ],
)
def test_cli_rejects_bad_csv_without_creating_outputs(files, content):
    files["input"].write_text(content, encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        predict.main(arguments(files))
    assert error.value.code == 2
    assert not files["output"].parent.exists()


@pytest.mark.parametrize("value", [b"1\x00invalid", b"\x00invalid"])
def test_cli_rejects_nul_bytes_before_parsing_without_outputs(files, monkeypatch, capsys, value):
    files["input"].write_bytes(
        b"sensor_000,sensor_001,sensor_002,sensor_003\n" + value + b",2,7,\n"
    )

    def forbid_parse(*args, **kwargs):
        raise AssertionError("NUL-containing input must not reach pandas.")

    monkeypatch.setattr(predict.pd, "read_csv", forbid_parse)
    with pytest.raises(SystemExit) as error:
        predict.main(arguments(files))
    assert error.value.code == 2
    assert "NUL" in capsys.readouterr().err
    assert not files["output"].parent.exists()


def test_cli_verifies_protocol_bytes_even_if_json_is_unchanged(files, capsys):
    files["protocol"].write_bytes(files["protocol"].read_bytes() + b"\n")
    with pytest.raises(SystemExit):
        predict.main(arguments(files))
    assert "Protocol file SHA256 differs" in capsys.readouterr().err
    assert not files["output"].parent.exists()


@pytest.mark.parametrize("change", ["schema", "candidate", "threshold", "source"])
def test_cli_checks_protocol_semantics_and_frozen_source(files, bundle, change, capsys):
    protocol = json.loads(files["protocol"].read_text())
    if change == "schema":
        protocol["schema_version"] = 2
    elif change == "candidate":
        protocol["selected_candidate"] = "dummy"
    elif change == "threshold":
        protocol["thresholds"][bundle["candidate"]] = 0.75
    else:
        protocol["source_sha256"] = "0" * 64
    files["protocol"].write_text(json.dumps(protocol), encoding="utf-8")
    bundle["protocol_sha256"] = sha256(files["protocol"].read_bytes()).hexdigest()
    joblib.dump(bundle, files["model"], compress=3)
    with pytest.raises(SystemExit):
        predict.main(arguments(files))
    message = capsys.readouterr().err
    assert (
        "Frozen source SHA256 differs" in message
        if change == "source"
        else "inconsistent" in message
    )
    assert not files["output"].parent.exists()
