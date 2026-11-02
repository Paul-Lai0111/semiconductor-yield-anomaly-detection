"""Exercise the run boundary using small real pipelines and isolated outputs."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

import secom.experiment as experiment
import secom.reporting as reporting
from secom.data import Dataset


def test_run_isolates_outputs_freezes_before_holdout_and_requires_explicit_overwrite(
    tmp_path, monkeypatch
):
    values, labels = make_classification(
        n_samples=200, n_features=8, n_informative=4, weights=[0.8, 0.2], random_state=5
    )
    frame = pd.DataFrame(values, columns=[f"sensor_{i:03d}" for i in range(8)])
    dataset = Dataset(frame, labels, pd.date_range("2008-01-01", periods=200), {})
    monkeypatch.setattr(experiment, "load_dataset", lambda _: dataset)
    figure_destinations = []
    monkeypatch.setattr(
        reporting, "create_figures", lambda path, *_: figure_destinations.append(path)
    )
    output = tmp_path / "trial"
    fitted_rows = []
    real_fit = experiment.fit_pipeline

    class ObservedPipeline:
        def __init__(self, pipeline):
            self.pipeline = pipeline

        def __getitem__(self, item):
            return self.pipeline[item]

        @property
        def named_steps(self):
            return self.pipeline.named_steps

        def predict_proba(self, X):
            split_file = output / "split_manifest.csv"
            split = pd.read_csv(split_file)
            holdout = set(split.loc[split.partition == "holdout", "row_id"])
            if holdout.intersection(X.index):
                frozen = json.loads((output / "protocol.json").read_text())
                assert frozen["selected_candidate"] in frozen["thresholds"]
                assert set(X.index) == holdout
            return self.pipeline.predict_proba(X)

    def observe_fit(name, X, y, *args):
        fitted_rows.append(set(X.index))
        return ObservedPipeline(real_fit(name, X, y, *args))

    monkeypatch.setattr(experiment, "fit_pipeline", observe_fit)
    # Wrapper is local to this regression test; serialize the real selected pipeline.
    real_dump = experiment.joblib.dump

    def dump_payload(payload, *args, **kwargs):
        return real_dump({**payload, "pipeline": payload["pipeline"].pipeline}, *args, **kwargs)

    monkeypatch.setattr(experiment.joblib, "dump", dump_payload)
    config = experiment.StudyConfig(folds=2, trees=2, bootstrap=3, temporal_check=False)
    result = experiment.run_study(config, tmp_path / "unused", output)
    split = pd.read_csv(output / "split_manifest.csv")
    development = set(split.loc[split.partition == "development", "row_id"])
    assert all(rows.issubset(development) for rows in fitted_rows)
    assert len(fitted_rows) == 7 * (config.folds + 1)
    assert figure_destinations == [output / "images"]
    assert (output / "artifacts" / "selected_model.joblib").is_file()
    assert not (tmp_path / "artifacts").exists()
    assert "across 2 overlapping training sets" in (output / "RESULTS.md").read_text()
    assert "(images/model_comparison.png)" in (output / "RESULTS.md").read_text()
    oof = pd.read_csv(output / "oof_predictions.csv")
    assert oof.groupby(["candidate", "row_id"]).size().eq(1).all()
    dev_metrics = pd.read_csv(output / "development_metrics.csv")
    assert (
        result["selected_candidate"]
        == dev_metrics.loc[dev_metrics.mean_fold_ap.idxmax(), "candidate"]
    )
    with pytest.raises(FileExistsError, match="--overwrite"):
        experiment.run_study(config, tmp_path / "unused", output)
    (output / "temporal_predictions.csv").write_text("stale predictions\n")
    experiment.run_study(config, tmp_path / "unused", output, overwrite=True)
    assert not (output / "temporal_predictions.csv").exists()
    np.testing.assert_array_equal(frame.to_numpy(), values)
