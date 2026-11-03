"""Build and execute an artifact-audit notebook for the completed chronological study."""

import os
import sys
import tempfile
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    markdown, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [
        markdown("""# SECOM: failure detection under an inspection budget

I compared 50 fixed base configurations and a predeclared rank blend to ask how many failures can be flagged when only 20% of a completed batch can be reviewed. The candidate set includes forests, boosting, support-vector methods, linear/discriminant models, nearest neighbors, and pass-only novelty detectors.

This notebook audits the completed run; it does not search another model or choose a cutoff on assessment labels. The latest time block was excluded from fitting and selection within this run. SECOM was explored previously, so this is not historically unseen or external evidence."""),
        code("""import json
from hashlib import sha256
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display
from sklearn.metrics import average_precision_score
from smartfab.policies import capacity_mask, evaluate_policy, threshold_mask

ROOT = Path.cwd()
OUT = ROOT / 'manufacturing_results'
metrics = json.loads((OUT / 'metrics.json').read_text())
protocol = json.loads((OUT / 'protocol.json').read_text())
selection = json.loads((OUT / 'selection.json').read_text())
assert sha256((OUT / 'protocol.json').read_bytes()).hexdigest() == metrics['protocol_sha256']
assert sha256((OUT / 'selection.json').read_bytes()).hexdigest() == metrics['selection_sha256']
for name, expected in protocol['source_sha256'].items():
    assert sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
print(f"Frozen winner: {selection['winner']}; {len(protocol['catalog'])} base configurations plus one rank blend.")
"""),
        markdown("""## 1. Fix the production question before choosing the model

The primary metric is failure recall at an exact per-block 20% review budget. All candidates share the same three expanding development windows; AP breaks ties. Native missing-value handling and missingness indicators are alternatives to imputation, and supervised feature selection is fitted only on training rows. Novelty detectors fit their preprocessing and model on passing training observations only.

The model-selection question concerns limited inspection resources. It does not assume that an alert prevents a defect or changes factory throughput."""),
        code("""manifest = pd.read_csv(OUT / 'split_manifest.csv', parse_dates=['timestamp'])
dev = manifest.loc[manifest.partition.eq('development')]
assessment = manifest.loc[manifest.partition.eq('temporal_assessment')]
assert set(dev.row_id).isdisjoint(assessment.row_id)
assert dev.timestamp.max() < assessment.timestamp.min()
display(manifest.groupby('partition').agg(rows=('row_id','size'), failures=('label','sum'), first=('timestamp','min'), last=('timestamp','max')))
display(Image(filename=str(OUT / 'figures/temporal_evaluation.png'), width=900))
"""),
        markdown("""## 2. Reconstruct the development selection

I selected by recorded forward-validation failures caught at the fixed review capacity, then mean block AP, then fixed catalog order. The blend members were determined on development data, and the blend has no fitted weights. Search statistics are optimistic after selecting among many configurations; they are not an independent performance estimate."""),
        code("""search = pd.read_csv(OUT / 'search_summary.csv')
forward = pd.read_csv(OUT / 'forward_predictions.csv', float_precision='round_trip')
assert len(search) == len(protocol['catalog']) + 1
assert search.iloc[0].candidate == selection['winner']
assert set(forward.row_id) <= set(dev.row_id)
assert not forward.duplicated(['candidate','row_id']).any()
for name, rows in forward.groupby('candidate'):
    caught, aps = 0, []
    for _, block in rows.groupby('block'):
        mask = capacity_mask(block.score, block.row_id, protocol['config']['review_fraction'], protocol['config']['seed'])
        caught += evaluate_policy(block.label, mask)['true_positives']
        aps.append(average_precision_score(block.label, block.score))
    saved = search.loc[search.candidate.eq(name)].iloc[0]
    assert caught == saved.true_positives
    np.testing.assert_allclose(np.mean(aps), saved.mean_block_ap)
display(search[['selection_rank','candidate','true_positives','failure_support','recall','mean_block_ap']].head(12))
display(Image(filename=str(OUT / 'figures/model_search.png'), width=1000))
"""),
        markdown("""## 3. Assess the frozen decision on later observations

The selected recipe and the historical v2 recipe are both refitted on the same earlier development observations. Each ranking receives the same number of reviews on the same latest block. The theoretical random expectation and its hypergeometric catch range describe chance allocations at that capacity; they are not an improvement confidence interval."""),
        code("""predictions = pd.read_csv(OUT / 'assessment_predictions.csv', float_precision='round_trip')
reconstructed = []
for name, rows in predictions.groupby('strategy',sort=False):
    assert set(rows.row_id) == set(assessment.row_id)
    mask = capacity_mask(rows.score, rows.row_id, protocol['config']['review_fraction'], protocol['config']['seed'])
    np.testing.assert_array_equal(mask, rows.review.astype(bool))
    result = evaluate_policy(rows.label, mask)
    saved = metrics['assessment'][name]['capacity']
    for key in ('true_positives','false_positives','false_negatives','true_negatives','alarm_count'):
        assert result[key] == saved[key]
    np.testing.assert_allclose(average_precision_score(rows.label,rows.score), metrics['assessment'][name]['average_precision'])
    reconstructed.append({'strategy':name,**result})
display(pd.DataFrame(reconstructed)[['strategy','alarm_count','true_positives','false_positives','false_negatives','precision','recall']])
display(Image(filename=str(OUT / 'figures/outcome_summary.png'), width=1100))
"""),
        code("""selected = metrics['assessment']['selected']['capacity']
reference = metrics['assessment']['v2_reference']['capacity']
delta = selected['true_positives'] - reference['true_positives']
display(Markdown(f"**Takeaway:** the selected method catches **{selected['true_positives']}/{selected['failure_support']}** failures with **{selected['alarm_count']}** reviews and **{selected['false_positives']}** false alerts. The same-budget difference versus v2 is **{delta:+d} failures**. This run does not establish a detection improvement when the difference is zero, and does not establish production benefits."))
"""),
        markdown("""## 4. Show the detection/workload frontier

The reported 5%, 10%, 20%, 30% and 50% budgets are descriptive policies on the already frozen rankings. The winner remains determined at 20%; a higher-recall budget is not relabeled as a better result without disclosing its additional reviews. The secondary F2 score threshold is selected from development OOF scores and cannot guarantee a future alert-rate cap."""),
        code("""budgets = pd.read_csv(OUT / 'assessment_budgets.csv')
display(budgets.loc[budgets.strategy.eq('selected'), ['review_fraction','alarm_count','true_positives','false_positives','false_negatives','precision','recall']])
selected_rows = predictions.loc[predictions.strategy.eq('selected')]
fixed = evaluate_policy(selected_rows.label, threshold_mask(selected_rows.score, selection['threshold_policy']['threshold']))
assert fixed == metrics['assessment']['selected']['fixed_threshold']
display(Image(filename=str(OUT / 'figures/detection_vs_capacity.png'), width=900))
"""),
        markdown("""## 5. Translate inspection decisions into explicit scenarios

The scenario model uses one normalized cost unit per review, assumed escape/review cost ratios, and assumed intervention effectiveness. It includes no additional review, full review, theoretical random review, and model-based review. An apparent simulated benefit is conditional on these assumptions and does not measure actual factory savings, prevented defects, yield, or throughput."""),
        code("""scenarios = pd.read_csv(OUT / 'cost_scenarios.csv')
assert np.isfinite(scenarios.simulated_total_cost_units).all()
assert scenarios.loc[scenarios.strategy.eq('random_same_capacity_expected'),'difference_vs_random'].eq(0).all()
display(Image(filename=str(OUT / 'figures/cost_sensitivity.png'), width=1000))
display(scenarios.loc[scenarios.escape_cost_per_review_cost.eq(20) & scenarios.assumed_intervention_effectiveness.eq(.5) & scenarios.passing_hold_cost_per_review_cost.eq(0), ['strategy','simulated_total_cost_units','difference_vs_random']])
"""),
        markdown("""## Completed conclusion

The study produces a bounded 50-configuration search, a predeclared blend, forward selection, a frozen temporal assessment, reconstructable review decisions, six explained figures, and a usable batch scoring interface. These artifacts quantify what this dataset supports and where the detection/workload tradeoff remains weak.

The assessment contains only 17 failures. Previously explored data, changing temporal composition, uncalibrated scores, anonymous sensors, and unverified decision-time feature availability limit claims of future reliability. The complete evidence is in [the generated report](manufacturing_results/RESULTS.md), with the exact procedure in [the protocol](docs/15_smart_manufacturing_protocol.md). No error-free detector or actual production-capacity increase is claimed."""),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3.12 (SECOM)",
        "language": "python",
        "name": "python3",
    }
    kernel = KernelManager(kernel_name="python3")
    kernel.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = NotebookClient(
        notebook,
        km=kernel,
        resources={"metadata": {"path": str(root)}},
        startup_timeout=300,
        timeout=300,
        record_timing=False,
    )
    with tempfile.TemporaryDirectory(prefix="smartfab-notebook-") as runtime:
        client.execute(
            cleanup_kc=True,
            env={**os.environ, "IPYTHONDIR": runtime, "JUPYTER_CONFIG_DIR": runtime},
        )
    nbformat.validate(notebook)
    target = root / "manufacturing_notebook.ipynb"
    nbformat.write(notebook, target)
    print(f"Executed and saved {target}")


if __name__ == "__main__":
    main()
