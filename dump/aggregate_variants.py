import argparse
import os
import pickle

import numpy as np

from config import cfg
from lib.dataset.hico import HicoSplit
from lib.dataset.utils import Splits
from lib.eval.evaluator_img import EvaluatorImg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('exp')
    args = parser.parse_args()
    model = args.model
    exp = args.exp

    runs, tr_results, zs_results = [], [], []
    for run in os.listdir(os.path.join('output', model)):
        if run.endswith(exp):
            print()
            print(run)
            cfg_file = os.path.join('output', model, run, 'config.pkl')
            cfg.load(file_path=cfg_file)

            assert not cfg.hicodet and cfg.seenf >= 0

            inds_dict = pickle.load(open(cfg.active_classes_file, 'rb'))
            pred_inds = sorted(inds_dict[Splits.TRAIN.value]['pred'].tolist())
            obj_inds = sorted(inds_dict[Splits.TRAIN.value]['obj'].tolist())

            splits = HicoSplit.get_splits(obj_inds=obj_inds, act_inds=pred_inds)
            train_split, val_split, test_split = splits[Splits.TRAIN], splits[Splits.VAL], splits[Splits.TEST]

            evaluator = EvaluatorImg(test_split)
            evaluator.load(cfg.eval_res_file)

            # Trained on:
            tr_metrics = evaluator.output_metrics(interactions_to_keep=sorted(train_split.active_interactions), no_print=True)
            tr_metrics = {f'tr_{k}': v for k, v in tr_metrics.items()}

            # Zero-shot
            unseen_interactions = set(range(train_split.full_dataset.num_interactions)) - set(train_split.active_interactions)
            zs_metrics = evaluator.output_metrics(interactions_to_keep=sorted(unseen_interactions), no_print=True)
            zs_metrics = {f'zs_{k}': v for k, v in zs_metrics.items()}

            runs.append(run)
            tr_results.append(np.mean(tr_metrics['tr_pM-mAP']) * 100)
            zs_results.append(np.mean(zs_metrics['zs_pM-mAP']) * 100)

    with open('aggr.pkl', 'wb') as f:
        pickle.dump([runs, tr_results, zs_results], f)

    print('\n' + '=' * 50, '\n')
    inds = np.argsort(np.array(zs_results))
    for i in inds:
        print(f'{runs[i]:40s}: {zs_results[i]:.2f}%')


if __name__ == '__main__':
    main()
