"""Small real-model inference, frozen recipes and safe batch output regressions."""

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from smartfab import predict
from smartfab.models import fit_candidate, predict_score
from smartfab.policies import blend_scores, capacity_mask


@pytest.fixture
def training():
    frame = pd.DataFrame(np.random.default_rng(17).normal(size=(24, 8)),
                         columns=[f"sensor_{i:03d}" for i in range(8)])
    frame.iloc[::3, 1] = np.nan
    frame.iloc[:, 6] = 7.0
    frame.iloc[:, 7] = np.nan
    return frame, np.tile([0, 1], 12)


@pytest.fixture
def bundle(training):
    spec = {"id": "tiny_rf", "family": "rf", "params": {"n_estimators": 2}, "novelty": False}
    return {
        "schema_version": 3, "winner": spec["id"], "is_rank_blend": False,
        "members": [{"spec": spec, "pipeline": fit_candidate(spec, *training, jobs=1)}],
        "sensor_columns": training[0].columns.tolist(), "review_fraction": 0.2,
        "seed": 2026, "secondary_threshold": 0.5,
    }


@pytest.fixture
def files(tmp_path, bundle, training):
    paths = {name: tmp_path / filename for name, filename in {
        "model": "model.joblib", "protocol": "protocol.json", "selection": "selection.json",
        "input": "input.csv", "output": "nested/predictions.csv",
        "metadata": "nested/predictions.metadata.json",
    }.items()}
    specs = [m["spec"] for m in bundle["members"]]
    protocol = {"schema_version": 3, "config": {
        "seed": 2026, "jobs": 2, "review_fraction": 0.2,
        "development_fraction": 0.8, "bootstrap": 2000,
    }, "source_sha256": predict.source_hashes(), "catalog": specs}
    paths["protocol"].write_text(json.dumps(protocol))
    bundle["protocol_sha256"] = sha256(paths["protocol"].read_bytes()).hexdigest()
    selection = {
        "winner": bundle["winner"], "members": [s["id"] for s in specs],
        "member_specs": specs, "is_rank_blend": False,
        "threshold_policy": {"threshold": 0.5}, "primary_review_fraction": 0.2,
        "protocol_sha256": bundle["protocol_sha256"],
    }
    paths["selection"].write_text(json.dumps(selection))
    bundle["selection_sha256"] = sha256(paths["selection"].read_bytes()).hexdigest()
    joblib.dump(bundle, paths["model"])
    training[0].iloc[:11, ::-1].to_csv(paths["input"], index=False)
    return paths


def arguments(files):
    return [v for name in ("model", "protocol", "selection", "input", "output")
            for v in (f"--{name}", str(files[name]))]


def test_scores_reorder_and_never_fit(bundle, training, monkeypatch):
    frame = training[0].iloc[:11].copy()
    member = bundle["members"][0]
    expected = predict_score(member["spec"], member["pipeline"], frame)
    state = joblib.hash(member["pipeline"])

    def forbid_fit(*args, **kwargs):
        raise AssertionError("Inference cannot fit.")

    for step in (member["pipeline"], *member["pipeline"].named_steps.values()):
        monkeypatch.setattr(step, "fit", forbid_fit)
    original = frame.copy()
    result = predict.predict_batch(bundle, frame.iloc[:, ::-1])
    np.testing.assert_array_equal(result.score, expected)
    assert result.columns.tolist() == ["unit_id", "score", "review", "threshold_alert"]
    assert result.review.sum() == 2
    np.testing.assert_array_equal(result.threshold_alert, expected >= 0.5)
    pd.testing.assert_frame_equal(frame, original)
    for step in (member["pipeline"], *member["pipeline"].named_steps.values()):
        monkeypatch.delattr(step, "fit")
    assert joblib.hash(member["pipeline"]) == state


def test_blend_uses_shared_exact_rank_helper(bundle, training):
    spec = {"id": "second_rf", "family": "rf", "params": {"n_estimators": 2}, "novelty": False}
    bundle["members"].append({"spec": spec, "pipeline": fit_candidate(spec, *training, seed=1, jobs=1)})
    bundle.update(winner="rank_blend_top2", is_rank_blend=True)
    frame = training[0].iloc[:11]
    expected = blend_scores([predict_score(m["spec"], m["pipeline"], frame) for m in bundle["members"]])
    np.testing.assert_array_equal(predict.predict_batch(bundle, frame).score, expected)
    # The second member's retained mask must also constrain each row.
    bundle["members"][1]["pipeline"]["sensors"].keep_mask_[:] = False
    bundle["members"][1]["pipeline"]["sensors"].keep_mask_[1] = True
    with pytest.raises(ValueError, match="observed retained sensor"):
        predict.predict_batch(bundle, frame)


def test_constant_ties_stable_ids_budget_floor_and_no_threshold(bundle, training):
    frame = pd.concat([training[0].iloc[[0]]] * 11, ignore_index=True)
    frame["unit_id"] = np.arange(100, 111)
    bundle["secondary_threshold"] = None
    result = predict.predict_batch(bundle, frame)
    assert result.review.sum() == 2
    assert not result.threshold_alert.any()
    np.testing.assert_array_equal(result.review, capacity_mask(result.score, result.unit_id, 0.2, 2026))
    reordered = predict.predict_batch(bundle, frame.iloc[::-1]).sort_values("unit_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(result, reordered)
    assert predict.predict_batch(bundle, frame.iloc[:4]).review.sum() == 0
    bundle["secondary_threshold"] = 0
    assert predict.predict_batch(bundle, frame).threshold_alert.sum() == 11


@pytest.mark.parametrize("change", ["missing", "label", "duplicate", "empty", "retained_missing"])
def test_invalid_schema_or_all_missing_retained_sensors(bundle, training, change):
    frame = training[0].iloc[:3].copy()
    if change == "missing":
        frame = frame.iloc[:, :-1]
    elif change == "label":
        frame["label"] = 0
    elif change == "duplicate":
        frame.columns = [frame.columns[0]] * 8
    elif change == "empty":
        frame = frame.iloc[:0]
    else:
        frame.iloc[0, :6] = np.nan
    with pytest.raises(ValueError):
        predict.predict_batch(bundle, frame)


@pytest.mark.parametrize("value", ["NaN", "inf", "1e999", "1+2j", 1 + 2j])
def test_nonfinite_or_nonreal_values_rejected(bundle, training, value):
    frame = training[0].iloc[:3].astype(object)
    frame.iloc[0, 0] = value
    with pytest.raises((ValueError, TypeError)):
        predict.predict_batch(bundle, frame)


def test_unit_ids_must_be_unique_integers_and_blank_strings_are_missing(bundle, training):
    frame = training[0].iloc[:3].copy()
    expected = predict.predict_batch(bundle, frame)
    strings = frame.astype(str).mask(frame.isna(), "")
    pd.testing.assert_frame_equal(expected, predict.predict_batch(bundle, strings))
    for ids in ([1, 1, 2], [1, 2, 3.5], ["1", "2", "3.0"]):
        with pytest.raises(ValueError, match="unique integers"):
            predict.predict_batch(bundle, frame.assign(unit_id=ids))


def test_module_cli_and_metadata_actual_hashes(files, bundle, training):
    result = subprocess.run([sys.executable, "-m", "smartfab.predict", *arguments(files)],
                            cwd=Path(predict.__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert "stable IDs are needed" in result.stdout
    expected = predict.predict_batch(bundle, training[0].iloc[:11])
    pd.testing.assert_frame_equal(pd.read_csv(files["output"]), expected)
    metadata = json.loads(files["metadata"].read_text())
    for name in ("model", "input", "protocol", "selection"):
        assert metadata[f"{name}_sha256"] == sha256(files[name].read_bytes()).hexdigest()
    assert metadata["source_sha256"] == predict.source_hashes()
    assert metadata["rows"] == 11 and metadata["review_budget"] == 2
    assert metadata["winner"] == bundle["winner"]
    assert metadata["unit_id_source"] == "batch_position"
    assert "Uncalibrated" in metadata["score_meaning"]
    assert "completed batch" in metadata["review_meaning"]
    assert "No capacity guarantee" in metadata["alert_meaning"]


@pytest.mark.parametrize("name", ["protocol", "selection"])
def test_recipe_hash_is_actual_bytes(files, name):
    files[name].write_bytes(files[name].read_bytes() + b"\n")
    with pytest.raises(SystemExit):
        predict.main(arguments(files))
    assert not files["output"].exists()


def test_recipe_semantics_and_source_inventory(files, bundle):
    protocol = files["protocol"].read_bytes()
    selection = json.loads(files["selection"].read_bytes())
    for key, value in (("winner", "other"), ("members", ["other"]),
                       ("threshold_policy", {"threshold": None}), ("primary_review_fraction", 0.3)):
        changed = json.dumps({**selection, key: value}).encode()
        bundle["selection_sha256"] = sha256(changed).hexdigest()
        with pytest.raises(ValueError, match="inconsistent"):
            predict.verify_recipe(bundle, protocol, changed)
    original_selection = files["selection"].read_bytes()
    bundle["selection_sha256"] = sha256(original_selection).hexdigest()
    recipe = json.loads(protocol)
    recipe["source_sha256"].pop("smartfab/predict.py")
    changed = json.dumps(recipe).encode()
    bundle["protocol_sha256"] = sha256(changed).hexdigest()
    selection["protocol_sha256"] = bundle["protocol_sha256"]
    selected = json.dumps(selection).encode()
    bundle["selection_sha256"] = sha256(selected).hexdigest()
    with pytest.raises(ValueError, match="source SHA256 differs"):
        predict.verify_recipe(bundle, changed, selected)


def test_csv_validation(files):
    header = files["input"].read_text().splitlines()[0]
    for content in (header.replace("sensor_001", "sensor_000") + "\n",
                    header + "\n1,2\n", header + "\n" + ",".join(["1"] * 8) + "\x00\n"):
        files["input"].write_text(content)
        with pytest.raises(SystemExit):
            predict.main(arguments(files))
        assert not files["output"].exists()


def test_existing_outputs_and_staging_failure(files, monkeypatch):
    files["output"].parent.mkdir()
    for key in ("output", "metadata"):
        files[key].write_text("existing\n")
    with pytest.raises(SystemExit):
        predict.main(arguments(files))
    real_temporary = predict.tempfile.NamedTemporaryFile
    calls = 0

    def fail_second_stage(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("cannot stage metadata")
        return real_temporary(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(predict.tempfile, "NamedTemporaryFile", fail_second_stage)
        with pytest.raises(SystemExit):
            predict.main([*arguments(files), "--overwrite"])
    for key in ("output", "metadata"):
        assert files[key].read_text() == "existing\n"
    assert set(files["output"].parent.iterdir()) == {files["output"], files["metadata"]}
    predict.main([*arguments(files), "--overwrite"])
    assert len(pd.read_csv(files["output"])) == 11


def test_path_collisions_and_aliases_even_with_overwrite(files):
    for source, target in (("output", "input"), ("output", "selection"),
                           ("input", "model"), ("selection", "protocol")):
        with pytest.raises(SystemExit):
            predict.main([*arguments({**files, source: files[target]}), "--overwrite"])
    files["output"].parent.mkdir()
    for method in ("symlink_to", "hardlink_to"):
        getattr(files["output"], method)(files["input"])
        with pytest.raises(SystemExit):
            predict.main([*arguments(files), "--overwrite"])
        files["output"].unlink()
