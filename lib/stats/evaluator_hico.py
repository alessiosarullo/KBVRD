from typing import List, Dict
import pickle

import numpy as np
from sklearn.metrics import average_precision_score

from lib.dataset.hico.hico_split import HicoSplit
from lib.models.containers import Prediction
from lib.stats.utils import Timer, sort_and_filter, MetricFormatter, BaseEvaluator


class HicoEvaluator(BaseEvaluator):
    def __init__(self, dataset_split: HicoSplit):
        super().__init__()

        self.dataset_split = dataset_split
        self.hico = dataset_split.full_dataset
        self.gt_scores = self.hico.split_annotations[self.dataset_split.split]
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
        self.metrics['M-mAP'] = average_precision_score(self.gt_scores, predict_hoi_scores)
        Timer.get('Eval epoch', 'Metrics').toc()

        Timer.get('Eval epoch').toc()

    def output_metrics(self, sort=False, actions_to_keep=None):
        mf = MetricFormatter()

        if actions_to_keep is not None:
            act_mask = np.zeros(self.hico.num_predicates, dtype=bool)
            act_mask[np.array(actions_to_keep).astype(np.int)] = True
            interaction_mask = act_mask[self.hico.interactions[:, 0]]
            interactions_to_keep = sorted(set(np.flatnonzero(interaction_mask).tolist()))
        else:
            interactions_to_keep = None

        gt_labels_vec = np.where(self.gt_scores)[1]
        gt_hoi_class_hist, hoi_metrics, hoi_class_inds = sort_and_filter(metrics=self.metrics,
                                                                         gt_labels=gt_labels_vec,
                                                                         all_classes=list(range(self.hico.num_interactions)),
                                                                         sort=sort,
                                                                         keep_inds=interactions_to_keep)
        mf.format_metric_and_gt_lines(gt_hoi_class_hist, hoi_metrics, hoi_class_inds, gt_str='GT HOIs')

        return [hoi_metrics]  # list is for compatibility
