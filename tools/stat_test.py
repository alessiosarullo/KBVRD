import argparse
import os

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import scipy.stats

# TODO


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fname')
    parser.add_argument('runs', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    fname = args.fname
    runs = args.runs

    print(fname)
    print(runs)

    summary_iterators = [EventAccumulator(os.path.join(p, 'tboard/Test')).Reload() for p in runs]
    tags = summary_iterators[0].Tags()['scalars']
    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    values_per_tag = {}
    steps = []
    for tag in tags:
        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            values_per_tag.setdefault(tag, []).append([e.value for e in events])
            assert len({e.step for e in events}) == 1
            if len(values_per_tag.keys()) == 1:
                steps.append(events[0].step)
    steps = np.array(steps)

    aggr_ops = {'mean': np.mean,
                'std': np.std}
    for aggr_op_name, aggr_op in aggr_ops.items():
        tblogger = SummaryWriter(os.path.join(fname, 'tboard/Test', aggr_op_name))
        for tag, values in values_per_tag.items():
            aggr_value_per_timestep = aggr_op(np.array(values), axis=1)
            for i, aggr_value in enumerate(aggr_value_per_timestep):
                tblogger.add_scalar(tag, aggr_value, global_step=steps[i])


if __name__ == '__main__':
    main()
