import argparse
import os

import numpy as np
import scipy.stats

from lib.stats.utils import get_runs_data


def group_reruns_and_ttest(names, summary_matrix, measures):
    rerun_flags = ['RE']
    name_to_idx = {name: i for i, name in enumerate(names)}
    unique_exps = {}
    for name in names:
        base_exp_name = name
        for i, m in enumerate(rerun_flags):
            if m in base_exp_name:
                for n in range(10):
                    rep_str = ' ' + m + str(n)
                    if rep_str in base_exp_name:
                        base_exp_name = base_exp_name.replace(rep_str, '')
                        break
                else:
                    base_exp_name = base_exp_name.replace(' ' + m, '')
        base_exp_name = base_exp_name.strip(' _-')
        unique_exps.setdefault(base_exp_name, []).append(name)

    new_names = []
    new_summary_matrix = np.zeros([len(unique_exps), summary_matrix.shape[1]], dtype=summary_matrix.dtype)
    for i, base in enumerate(unique_exps.keys()):
        reruns = unique_exps[base]
        if len(reruns) > 1:
            # print('%-55s' % base, '=', reruns)
            new_names.append(base + ' [%d]' % len(reruns))
        else:
            new_names.append(base)
        new_summary_matrix[i, :] = np.mean(summary_matrix[np.array([name_to_idx[r] for r in reruns]), :], axis=0)

    # Welch’s t-test
    print('=' * 100)
    compare_with = [
        ('IMP pretrained', 'IMP sig CSL 1bg norpn'),
        ('IMP pretrained', 'IMP sig CSL fgpred BG'),
        ('NMOTIFS pretrained', 'NM sig CSL 1bg'),
        ('NMOTIFS pretrained', 'NM sig CSL blcbg fgpred'),
        ('P2G vanilla50', 'P2G sig50 CSL'),
        ('P2G vanilla50', 'P2G sig50 CSL fgpred')
    ]
    for baseline, mod in compare_with:
        baseline_stats = summary_matrix[np.array([name_to_idx[r] for r in unique_exps[baseline]]), :]
        mod_stats = summary_matrix[np.array([name_to_idx[r] for r in unique_exps[mod]]), :]
        pvalues = []
        for c in range(summary_matrix.shape[1]):
            if 'pretrained' in baseline:
                assert baseline_stats.shape[0] == 1
                pvalue = scipy.stats.ttest_1samp(mod_stats[:, c], popmean=baseline_stats[0, c])[1]
            else:
                pvalue = scipy.stats.ttest_ind(baseline_stats[:, c], mod_stats[:, c], equal_var=False)[1]
            pvalues.append(pvalue)
        print('%s [%d] vs [%d] %s' % (baseline, baseline_stats.shape[0], mod_stats.shape[0], mod))
        print(' '.join(['%10s' % m for m in measures]))
        print(' '.join(['%10.2e' % p for p in pvalues]))
        print()
    print('=' * 100)
    return new_names, new_summary_matrix


def get_stats_on_best_results(runs, exp_data):
    # Result obtained at the lowest validation action loss.
    assert np.all(exp_data['Val']['steps'] == exp_data['Test']['steps'])
    best_val_loss_step_per_run = np.argmin(exp_data['Val']['values']['Act_loss'], axis=1)
    test_data = exp_data['Test']['values']['zs_pM-mAP']
    test_accuracy_per_run = test_data[np.arange(test_data.shape[0]), best_val_loss_step_per_run]

    sp = max([len(r) for r in runs])
    for i, r in enumerate(runs):
        print(f'{r:{sp}s} {test_accuracy_per_run[i] * 100:6.3f}')
    print(f'{"Mean":>{sp}s} {np.mean(test_accuracy_per_run) * 100:6.3f}')
    print(f'{"Std":>{sp}s} {np.std(test_accuracy_per_run) * 100:6.3f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    args = parser.parse_args()
    dir = args.dir

    print(dir)
    runs = []
    for run_dir in os.listdir(dir):
        if 'RUN' in run_dir:
            runs.append(os.path.join(dir, run_dir))
    runs = sorted(runs)

    exp_data = get_runs_data(runs)
    get_stats_on_best_results(runs, exp_data)


if __name__ == '__main__':
    main()
