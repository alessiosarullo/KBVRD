import pickle
from typing import List, Dict

import numpy as np
from sklearn.metrics import average_precision_score

from lib.dataset.hico.hico_split import HicoSplit
from lib.models.containers import Prediction
from lib.stats.utils import Timer, sort_and_filter, MetricFormatter, BaseEvaluator


class HicoEvaluator(BaseEvaluator):
    def __init__(self, dataset_split: HicoSplit):
        super().__init__()

        self.dataset_split = dataset_split
        self.full_dataset = dataset_split.full_dataset
        self.gt_scores = self.full_dataset.split_annotations[self.dataset_split.split]
        self.metrics = {}  # type: Dict[str, np.ndarray]

    def save(self, fn):
        with open(fn, 'wb') as f:
            pickle.dump({'metrics': self.metrics}, f)

    def evaluate_predictions(self, predictions: List[Dict]):
        assert len(predictions) == self.dataset_split.num_images, (len(predictions), self.dataset_split.num_images)

        Timer.get('Eval epoch').tic()
        Timer.get('Eval epoch', 'Predictions').tic()
        predict_hoi_scores = np.full_like(self.gt_scores, fill_value=np.nan)
        for i, res in enumerate(predictions):
            prediction = Prediction(res)
            predict_hoi_scores[i, :] = prediction.hoi_scores
        Timer.get('Eval epoch', 'Predictions').toc()

        Timer.get('Eval epoch', 'Metrics').tic()
        gt_scores = self.gt_scores
        gt_scores[gt_scores < 0] = 0
        self.metrics['M-mAP'] = average_precision_score(gt_scores, predict_hoi_scores, average=None)
        Timer.get('Eval epoch', 'Metrics').toc()

        Timer.get('Eval epoch').toc()

    def output_metrics(self, sort=False, interactions_to_keep=None, actions_to_keep=None, no_print=False):
        mf = MetricFormatter()

        if interactions_to_keep is None and actions_to_keep is not None:
            act_mask = np.zeros(self.full_dataset.num_actions, dtype=bool)
            act_mask[np.array(actions_to_keep).astype(np.int)] = True
            interaction_mask = act_mask[self.full_dataset.interactions[:, 0]]
            interactions_to_keep = sorted(np.flatnonzero(interaction_mask).tolist())
        metrics = self._output_metrics(mf, sort=sort, interactions_to_keep=interactions_to_keep, no_print=no_print)

        # Same, but with null interaction filtered
        actions_to_keep = sorted(set(actions_to_keep or range(self.full_dataset.num_actions)) - {0})
        act_mask = np.zeros(self.full_dataset.num_actions, dtype=bool)
        act_mask[np.array(actions_to_keep).astype(np.int)] = True
        no_null_interaction_mask = act_mask[self.full_dataset.interactions[:, 0]]
        if interactions_to_keep is None:
            interactions_to_keep = sorted(np.flatnonzero(no_null_interaction_mask).tolist())
        else:
            interaction_mask = np.zeros(self.full_dataset.num_interactions, dtype=bool)
            interaction_mask[np.array(interactions_to_keep).astype(np.int)] = True
            interactions_to_keep = sorted(np.flatnonzero(no_null_interaction_mask & interaction_mask).tolist())
        pos_metrics = self._output_metrics(mf, sort=sort, interactions_to_keep=interactions_to_keep, no_print=no_print)

        for k, v in pos_metrics.items():
            assert k in metrics.keys()
            metrics[f'p{k}'] = v
        return metrics

    def _output_metrics(self, mf: MetricFormatter, sort, interactions_to_keep, no_print):
        gt_labels_vec = np.where(self.gt_scores)[1]
        gt_hoi_class_hist, hoi_metrics, hoi_class_inds = sort_and_filter(metrics=self.metrics,
                                                                         gt_labels=gt_labels_vec,
                                                                         all_classes=list(range(self.full_dataset.num_interactions)),
                                                                         sort=sort,
                                                                         keep_inds=interactions_to_keep)
        if not no_print:
            mf.format_metric_and_gt_lines(gt_hoi_class_hist, hoi_metrics, hoi_class_inds, gt_str='GT HOIs')
        return hoi_metrics
