import argparse
import os

import numpy as np
import scipy.stats

from lib.utils import get_runs_data, rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir1')
    parser.add_argument('dir2')
    parser.add_argument('measure')
    args = parser.parse_args()
    dir1 = args.dir1
    dir2 = args.dir2
    measure = args.measure

    print(dir1, dir2)
    runs_sets = []
    for wd in [dir1, dir2]:
        runs_wd = []
        for run_dir in os.listdir(wd):
            if 'RUN' in run_dir:
                runs_wd.append(os.path.join(wd, run_dir))
        runs_wd = sorted(runs_wd)
        runs_sets.append(runs_wd)

    all_exp_data = [get_runs_data(runs) for runs in runs_sets]

    # Result obtained at the lowest validation action loss.
    test_accs = []
    for exp_data in all_exp_data:
        test_data = exp_data['Test']['values'][measure]
        # val_losses = np.stack([v for k, v in exp_data['Val']['values'].items() if 'Act' in k and 'loss' in k], axis=0)
        val_losses = exp_data['Val']['values']['Act_loss']

        if np.all(exp_data['Val']['steps'] == exp_data['Test']['steps']):
            val_losses = val_losses
        else:
            print(f"{exp_data['Val']['steps'].size} val steps, but only {exp_data['Test']['steps'].size} test steps.")
            val_steps = exp_data['Val']['steps']
            test_steps = exp_data['Test']['steps']
            valid_val_steps_inds = []
            for ts in test_steps:
                inds = np.flatnonzero(val_steps == ts)
                valid_val_steps_inds.append(inds.item())
            valid_val_steps_inds = np.array(valid_val_steps_inds)
            val_losses = val_losses[..., valid_val_steps_inds]

        best_val_loss_step_per_run = np.argmin(val_losses, axis=1)
        # best_val_loss_step_per_run = np.argmin(rank(val_losses).mean(axis=0), axis=1)

        test_accuracy_per_run = test_data[np.arange(test_data.shape[0]), best_val_loss_step_per_run]
        sp = max([len(r) for r in runs_sets])
        print(f'{"Mean":>{sp}s} {np.mean(test_accuracy_per_run):8.5f}')
        print(f'{"Std":>{sp}s} {np.std(test_accuracy_per_run):8.5f}')
        test_accs.append(test_accuracy_per_run)

    # Welch’s t-test
    pvalue = scipy.stats.ttest_ind(test_accs[0], test_accs[1], equal_var=False)[1]
    print(f'{measure:>15s}')
    print(f'p = {pvalue:11.2e}')


if __name__ == '__main__':
    main()
