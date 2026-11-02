"""Keep supplementary checks descriptive, capacity bounded, and development only."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

import secom.experiment as experiment
from scripts.development_diagnostics import (
    STABILITY_CANDIDATES,
    capacity_tables,
    repeated_comparisons,
    review_mask,
    run_diagnostics,
    validate_evidence,
)
from secom.data import Dataset


def test_review_budget_never_splits_a_tie_or_skips_a_higher_group():
    scores = [0.9, 0.9, 0.9, 0.7, 0.5]
    assert not review_mask(scores, 0.4).any()
    np.testing.assert_array_equal(review_mask(scores, 0.6), [True, True, True, False, False])


def test_capacity_is_order_invariant_and_uses_unrounded_scores():
    scores = np.array([0.8, 0.800001, 0.7, 0.2, 0.800001])
    expected = review_mask(scores, 0.4)
    np.testing.assert_array_equal(expected, [False, True, False, False, True])
    order = np.array([3, 2, 4, 1, 0])
    np.testing.assert_array_equal(review_mask(scores[order], 0.4), expected[order])


@pytest.mark.parametrize("fraction,expected", [(0, 0), (0.99, 0), (1, 10)])
def test_constant_scores_require_room_for_the_whole_group(fraction, expected):
    assert review_mask(np.ones(10), fraction).sum() == expected


@pytest.mark.parametrize("scores", [[], [[0.1, 0.2]], [np.nan], [np.inf]])
def test_invalid_capacity_scores_are_rejected(scores):
    with pytest.raises(ValueError, match="nonempty finite one-dimensional"):
        review_mask(scores, 0.1)


@pytest.mark.parametrize("budget", [-0.1, 1.01, np.nan, np.inf])
def test_invalid_capacity_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="fraction"):
        review_mask([0.1, 0.2], budget)


def test_capacity_aggregates_counts_and_floors_each_block():
    frame = pd.DataFrame(
        {
            "candidate": ["rf_knn"] * 8,
            "fold": [0] * 5 + [1] * 3,
            "label": [1, 0, 1, 0, 0, 0, 1, 1],
            "score": [0.9, 0.8, 0.7, 0.3, 0.2, 0.9, 0.8, 0.7],
        }
    )
    blocks, summary = capacity_tables(frame, [0.5])
    row = summary.iloc[0]
    assert blocks.maximum_reviews.tolist() == [2, 1]
    assert row.maximum_reviews == row.alarm_count == 3
    assert row.true_positives == 1
    assert row.false_positives == 2
    assert row.false_negatives == 3
    assert row.precision == pytest.approx(1 / 3)
    assert row.recall == pytest.approx(1 / 4)
    assert row.precision != blocks.precision.mean()


def test_stability_fits_and_predicts_only_saved_development_rows(monkeypatch):
    values, labels = make_classification(
        n_samples=160, n_features=8, n_informative=4, weights=[0.7, 0.3], random_state=7
    )
    frame = pd.DataFrame(values, columns=[f"sensor_{i:03d}" for i in range(8)])
    dataset = Dataset(frame, labels, pd.date_range("2008-01-01", periods=160), {})
    development = np.arange(120)
    fitted = []
    original = experiment.fit_pipeline

    class ObservedModel:
        def __init__(self, model):
            self.model = model

        def predict_proba(self, X):
            assert set(X.index) <= set(development)
            return self.model.predict_proba(X)

        def __getitem__(self, key):
            return self.model[key]

        @property
        def named_steps(self):
            return self.model.named_steps

    def observe_fit(name, X, y, *args):
        assert set(X.index) <= set(development)
        fitted.append(set(X.index))
        return ObservedModel(original(name, X, y, *args))

    monkeypatch.setattr(experiment, "fit_pipeline", observe_fit)
    summary, folds, predictions, assignments = repeated_comparisons(
        dataset, development[::-1], (17, 43), folds=2, trees=2, jobs=1
    )
    assert len(fitted) == len(STABILITY_CANDIDATES) * 2 * 2
    assert len(summary) == 8 and len(folds) == 16
    assert predictions.groupby(["candidate", "repeat", "row_id"]).size().eq(1).all()
    assert set(predictions.row_id) == set(development)
    assert len(assignments) == 2 * len(development)
    assert predictions.groupby(["repeat", "row_id"]).fold.nunique().eq(1).all()
    assert summary.oof_threshold.between(0, 1).all()
    assert summary.rank_among_four.between(1, 4).all()
    np.testing.assert_array_equal(dataset.X, values)


@pytest.fixture
def saved_evidence(tmp_path):
    original = Path(__file__).resolve().parents[1] / "results"
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for name in ("protocol.json", "metrics.json", "split_manifest.csv", "oof_predictions.csv"):
        (evidence / name).write_bytes((original / name).read_bytes())
    manifest = pd.read_csv(evidence / "split_manifest.csv")
    metrics = json.loads((evidence / "metrics.json").read_text())
    dataset = SimpleNamespace(
        y=manifest.label.to_numpy(), manifest={"sha256": metrics["dataset"]["sha256"]}
    )
    return evidence, dataset


def test_saved_membership_and_folds_reproduce_the_frozen_protocol(saved_evidence):
    evidence, dataset = saved_evidence
    _, development, oof = validate_evidence(evidence, dataset)
    assert len(development) == 1253
    assert development.label.sum() == 83
    assert set(oof.row_id) == set(development.index)


def test_consistent_csv_swap_cannot_move_a_holdout_row_into_development(saved_evidence):
    evidence, dataset = saved_evidence
    manifest = pd.read_csv(evidence / "split_manifest.csv")
    dev = manifest.loc[(manifest.partition == "development") & (manifest.label == 0)].iloc[0]
    holdout = manifest.loc[(manifest.partition == "holdout") & (manifest.label == 0)].iloc[0]
    manifest.loc[manifest.row_id == dev.row_id, ["partition", "validation_fold"]] = ["holdout", -1]
    manifest.loc[manifest.row_id == holdout.row_id, ["partition", "validation_fold"]] = [
        "development",
        dev.validation_fold,
    ]
    manifest.to_csv(evidence / "split_manifest.csv", index=False)
    oof = pd.read_csv(evidence / "oof_predictions.csv", float_precision="round_trip")
    oof.loc[oof.row_id == dev.row_id, "row_id"] = holdout.row_id
    oof.to_csv(evidence / "oof_predictions.csv", index=False)
    with pytest.raises(ValueError, match="frozen outer split"):
        validate_evidence(evidence, dataset)


def test_altered_validation_assignment_is_rejected(saved_evidence):
    evidence, dataset = saved_evidence
    manifest = pd.read_csv(evidence / "split_manifest.csv")
    mask = manifest.partition == "development"
    manifest.loc[mask, "validation_fold"] = (manifest.loc[mask, "validation_fold"] + 1) % 5
    manifest.to_csv(evidence / "split_manifest.csv", index=False)
    with pytest.raises(ValueError, match="frozen fold assignments"):
        validate_evidence(evidence, dataset)


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_overwrite_cannot_write_through_a_link_to_original_metrics(tmp_path, kind):
    evidence, output = tmp_path / "original", tmp_path / "supplement"
    evidence.mkdir()
    output.mkdir()
    source = evidence / "metrics.json"
    original = b'{"keep": "original measurements"}\n'
    source.write_bytes(original)
    destination = output / "provenance.json"
    if kind == "symlink":
        destination.symlink_to(source)
    else:
        destination.hardlink_to(source)
    with pytest.raises(ValueError, match="ordinary unlinked files"):
        run_diagnostics(evidence, tmp_path / "unused", output, overwrite=True)
    assert source.read_bytes() == original
