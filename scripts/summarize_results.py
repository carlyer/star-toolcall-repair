"""Paired bootstrap over unseen prompt groups, with separate call/no-call metrics."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np


def group_values(rows, metric):
    groups = collections.defaultdict(list)
    for row in rows:
        if metric == 'tool_exact_match' and row['tool_case']:
            value = int(row['exact'])
        elif metric == 'no_tool_decision_accuracy' and not row['tool_case']:
            value = int(row['format_valid'] and not row['predicted_tool_call'])
        else:
            continue
        groups[row['prompt_sha256']].append(value)
    return {key: float(np.mean(values)) for key, values in groups.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--experiment', default='generic-teacher-hammer')
    args = parser.parse_args()
    root = Path(args.run_dir)
    reports, predictions = {}, {}
    for stage in ['base', 'ckd', 'sim-rl']:
        reports[stage] = json.loads((root/f'eval-{stage}.json').read_text())
        predictions[stage] = [json.loads(line) for line in (root/f'eval-{stage}.predictions.jsonl').open()]
    identities = [(row['id'], row['prompt_sha256']) for row in predictions['base']]
    assert all([(row['id'], row['prompt_sha256']) for row in rows] == identities for rows in predictions.values())
    comparisons = {}
    for stage in ['ckd', 'sim-rl']:
        compared = {}
        bootstraps = {}
        rng = np.random.default_rng(42)
        for metric in ['tool_exact_match', 'no_tool_decision_accuracy']:
            baseline = group_values(predictions['base'], metric)
            current = group_values(predictions[stage], metric)
            assert baseline.keys() == current.keys()
            if not baseline:
                compared[metric] = {'delta': None, 'reason': 'No examples of this class in held-out data'}
                continue
            keys = sorted(baseline)
            delta = np.asarray([current[key]-baseline[key] for key in keys])
            indices = rng.integers(0, len(keys), (10000, len(keys)))
            samples = delta[indices].mean(axis=1)
            bootstraps[metric] = samples
            compared[metric] = {'delta': float(delta.mean()), 'paired_bootstrap_95pct_interval': np.quantile(samples, [.025, .975]).tolist(), 'unique_prompt_groups': len(keys)}
        if len(bootstraps) == 2:
            samples = (bootstraps['tool_exact_match']+bootstraps['no_tool_decision_accuracy'])/2
            compared['macro_policy_accuracy'] = {'delta': (compared['tool_exact_match']['delta']+compared['no_tool_decision_accuracy']['delta'])/2, 'paired_bootstrap_95pct_interval': np.quantile(samples, [.025, .975]).tolist()}
        comparisons[stage+'_minus_base'] = compared
    output = {'experiment': args.experiment, 'original_teacher_and_full_data_reproduced': False, 'official_paper_benchmark': False, 'heldout_rows': len(identities), 'evaluations': reports, 'comparisons': comparisons, 'bootstrap_seed': 42, 'bootstrap_resamples': 10000, 'bootstrap_scope': 'held-out prompt groups; does not estimate variability across training seeds'}
    (root/'evaluation-summary.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
