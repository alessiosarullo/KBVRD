from collections import Counter
from typing import List, Dict

import pickle
import numpy as np

from lib.bbox_utils import compute_ious
from lib.dataset.hicodet.hicodet_split import HicoDetSplit, Example
from lib.models.containers import Prediction
from lib.stats.utils import Timer, sort_and_filter, MetricFormatter, BaseEvaluator


class Evaluator(BaseEvaluator):
    def __init__(self, dataset_split: HicoDetSplit, iou_thresh=0.5, hoi_score_thr=None, num_hoi_thr=None):
        super().__init__()
        self.iou_thresh = iou_thresh
        self.hoi_score_thr = hoi_score_thr
        self.num_hoi_thr = num_hoi_thr

        self.dataset_split = dataset_split
        self.full_dataset = dataset_split.full_dataset
        self._init()

    def _init(self):
        self.gt_hoi_classes = []
        self.predict_hoi_scores = []
        self.pred_gt_assignment_per_hoi = []
        self.gt_hit_per_prediction = {}
        self.gt_hit_per_prediction2 = []
        self.gt_count = 0

        self.metrics = {}  # type: Dict[str, np.ndarray]

    def save(self, fn):
        with open(fn, 'wb') as f:
            pickle.dump({'hits': self.gt_hit_per_prediction,
                         'gt_classes': np.concatenate(self.gt_hoi_classes, axis=0),
                         'metrics': self.metrics}, f)

    def evaluate_predictions(self, predictions: List[Dict]):
        self._init()
        assert len(predictions) == self.dataset_split.num_images, (len(predictions), self.dataset_split.num_images)

        Timer.get('Eval epoch').tic()
        Timer.get('Eval epoch', 'Predictions').tic()
        for i, res in enumerate(predictions):
            ex = self.dataset_split.get_img_entry(i, read_img=False)
            prediction = Prediction(res)
            self.match_prediction_to_gt(i, ex, prediction)
        Timer.get('Eval epoch', 'Predictions').toc()
        Timer.get('Eval epoch', 'Metrics').tic()
        self.compute_metrics()
        Timer.get('Eval epoch', 'Metrics').toc()
        Timer.get('Eval epoch').toc()

    def compute_metrics(self):
        gt_hoi_classes = np.concatenate(self.gt_hoi_classes, axis=0)
        assert self.gt_count == gt_hoi_classes.shape[0]
        predict_hoi_scores = np.concatenate(self.predict_hoi_scores, axis=0)
        pred_gt_ho_assignment = np.concatenate(self.pred_gt_assignment_per_hoi, axis=0)

        gt_hoi_classes_count = Counter(gt_hoi_classes.tolist())

        ap = np.zeros(self.full_dataset.num_interactions)
        recall = np.zeros(self.full_dataset.num_interactions)
        for j in range(self.full_dataset.num_interactions):
            num_gt_hois = gt_hoi_classes_count[j]
            if num_gt_hois == 0:
                continue

            p_hoi_scores = predict_hoi_scores[:, j]
            p_gt_ho_assignment = pred_gt_ho_assignment[:, j]
            if self.hoi_score_thr is not None:
                inds = (p_hoi_scores >= self.hoi_score_thr)
                p_hoi_scores = p_hoi_scores[inds]
                p_gt_ho_assignment = p_gt_ho_assignment[inds]
            if self.num_hoi_thr is not None:
                p_hoi_scores = p_hoi_scores[:self.num_hoi_thr]
                p_gt_ho_assignment = p_gt_ho_assignment[:self.num_hoi_thr]
            rec_j, prec_j, ap_j = self.eval_single_interaction_class(p_hoi_scores, p_gt_ho_assignment, num_gt_hois)
            ap[j] = ap_j
            if rec_j.size > 0:
                recall[j] = rec_j[-1]

        self.metrics['M-mAP'] = ap
        self.metrics['M-rec'] = recall

    def output_metrics(self, sort=False, actions_to_keep=None, return_labels=False, print_out=True):
        mf = MetricFormatter()

        gt_hoi_triplets = self.dataset_split.hoi_triplets
        orig_gt_hoi_labels = self.full_dataset.op_pair_to_interaction[gt_hoi_triplets[:, 2], gt_hoi_triplets[:, 1]]
        metrics = {k: v for k, v in self.metrics.items()}
        gt_hoi_labels, interactions_to_keep = self.filter_actions(orig_gt_hoi_labels, actions_to_keep)
        assert np.all(gt_hoi_labels >= 0)
        gt_hoi_class_hist, metrics, hoi_class_inds = sort_and_filter(metrics=metrics,
                                                                     gt_labels=gt_hoi_labels,
                                                                     all_classes=list(range(self.full_dataset.num_interactions)),
                                                                     sort=sort,
                                                                     keep_inds=interactions_to_keep)
        mf.format_metric_and_gt_lines(gt_hoi_class_hist, metrics, hoi_class_inds, gt_str='GT HOIs')

        # Same, but with null interaction filtered
        actions_to_keep = sorted(set(actions_to_keep or range(self.full_dataset.num_actions)) - {0})
        pos_gt_hoi_labels, interactions_to_keep = self.filter_actions(orig_gt_hoi_labels, actions_to_keep)
        assert np.all(pos_gt_hoi_labels >= 0)
        pos_metrics = {f'p{k}': v for k, v in self.metrics.items()}
        gt_hoi_class_hist, pos_metrics, hoi_class_inds = sort_and_filter(metrics=pos_metrics,
                                                                         gt_labels=pos_gt_hoi_labels,
                                                                         all_classes=list(range(self.full_dataset.num_interactions)),
                                                                         sort=sort,
                                                                         keep_inds=interactions_to_keep)
        mf.format_metric_and_gt_lines(gt_hoi_class_hist, pos_metrics, hoi_class_inds, gt_str='GT HOIs', print_out=print_out)

        for k, v in pos_metrics.items():
            assert k not in metrics.keys()
            metrics[k] = v
        if return_labels:
            return metrics, self.dataset_split.obj_labels, gt_hoi_labels
        return metrics

    def filter_actions(self, gt_hoi_labels, actions_to_keep):
        if actions_to_keep is not None:
            act_mask = np.zeros(self.full_dataset.num_actions, dtype=bool)
            act_mask[np.array(actions_to_keep).astype(np.int)] = True
            interaction_mask = act_mask[self.full_dataset.interactions[:, 0]]
            gt_hoi_labels = gt_hoi_labels[interaction_mask[gt_hoi_labels]]
            interactions_to_keep = set(np.flatnonzero(interaction_mask).tolist())
        else:
            interactions_to_keep = None
        return gt_hoi_labels, interactions_to_keep

    def match_prediction_to_gt(self, im_id, gt_entry: Example, prediction: Prediction):
        if isinstance(gt_entry, Example):
            gt_hoi_triplets = gt_entry.gt_hois[:, [0, 2, 1]]  # (h, o, i)
            num_gt_hois = gt_hoi_triplets.shape[0]

            gt_boxes = gt_entry.gt_boxes.astype(np.float, copy=False)

            gt_hoi_classes = self.full_dataset.op_pair_to_interaction[gt_entry.gt_obj_classes[gt_hoi_triplets[:, 1]], gt_hoi_triplets[:, 2]]
            assert np.all(gt_hoi_classes) >= 0

            gt_ho_ids = self.gt_count + np.arange(num_gt_hois)
            self.gt_count += num_gt_hois
        else:
            raise ValueError('Unknown type for GT entry: %s.' % str(type(gt_entry)))

        predict_hoi_scores = np.zeros([0, self.full_dataset.num_interactions])
        predict_ho_pairs = np.zeros((0, 2), dtype=np.int)
        predict_boxes = np.zeros((0, 4))
        if prediction.obj_boxes is not None:
            assert prediction.obj_im_inds.shape[0] == prediction.obj_boxes.shape[0]

            predict_boxes = prediction.obj_boxes

            if prediction.ho_pairs is not None:
                assert len(np.unique(prediction.obj_im_inds)) == len(np.unique(prediction.ho_img_inds)) == 1

                predict_ho_pairs = prediction.ho_pairs
                predict_hoi_scores = prediction.hoi_scores
                if predict_hoi_scores is None:
                    assert prediction.obj_im_inds.shape[0] == prediction.obj_scores.shape[0]
                    assert prediction.action_scores is not None

                    if len(self.dataset_split.active_actions) < self.full_dataset.num_actions:
                        predict_action_scores = np.zeros((prediction.action_scores.shape[0], self.full_dataset.num_actions))
                        predict_action_scores[:, self.dataset_split.active_actions] = prediction.action_scores
                    else:
                        predict_action_scores = prediction.action_scores
                    predict_obj_scores_per_ho_pair = prediction.obj_scores[predict_ho_pairs[:, 1], :]

                    predict_hoi_scores = np.empty([predict_ho_pairs.shape[0], self.full_dataset.num_interactions])
                    for iid, (pid, oid) in enumerate(self.full_dataset.interactions):
                        predict_hoi_scores[:, iid] = predict_obj_scores_per_ho_pair[:, oid] * predict_action_scores[:, pid]
        else:
            assert prediction.ho_pairs is None

        pred_gt_ious = compute_ious(predict_boxes, gt_boxes)
        pred_gt_assignment_per_hoi = np.full((predict_hoi_scores.shape[0], self.full_dataset.num_interactions), fill_value=-1, dtype=np.int)
        for predict_idx, (ph, po) in enumerate(predict_ho_pairs):
            gt_pair_ious = np.zeros(num_gt_hois)
            for gtidx, (gh, go, gi) in enumerate(gt_hoi_triplets):
                iou_h = pred_gt_ious[ph, gh]
                iou_o = pred_gt_ious[po, go]
                gt_pair_ious[gtidx] = min(iou_h, iou_o)
            if np.any(gt_pair_ious >= self.iou_thresh):
                gt_pair_ious_per_hoi = np.zeros((num_gt_hois, self.full_dataset.num_interactions))
                gt_pair_ious_per_hoi[np.arange(num_gt_hois), gt_hoi_classes] = gt_pair_ious
                gt_assignments = gt_pair_ious_per_hoi.argmax(axis=0)[np.any(gt_pair_ious_per_hoi >= self.iou_thresh, axis=0)]
                gt_hoi_assignments = gt_hoi_classes[gt_assignments]
                assert np.unique(gt_assignments).size == gt_assignments.size
                assert np.unique(gt_hoi_assignments).size == gt_hoi_assignments.size
                pred_gt_assignment_per_hoi[predict_idx, gt_hoi_assignments] = gt_ho_ids[gt_assignments]

        self.gt_hoi_classes.append(gt_hoi_classes)
        self.predict_hoi_scores.append(predict_hoi_scores)
        self.pred_gt_assignment_per_hoi.append(pred_gt_assignment_per_hoi)

    def eval_single_interaction_class(self, predicted_conf_scores, pred_gtid_assignment, num_hoi_gt_positives):
        num_predictions = predicted_conf_scores.shape[0]
        tp = np.zeros(num_predictions)

        if num_predictions > 0:
            inds = np.argsort(predicted_conf_scores)[::-1]
            pred_gtid_assignment = pred_gtid_assignment[inds]

            matched_gt_inds, highest_scoring_pred_idx_per_gt_ind = np.unique(pred_gtid_assignment, return_index=True)
            if matched_gt_inds[0] == -1:
                matched_gt_inds = matched_gt_inds[1:]
                highest_scoring_pred_idx_per_gt_ind = highest_scoring_pred_idx_per_gt_ind[1:]
            tp[highest_scoring_pred_idx_per_gt_ind] = 1

            # Timer.get('Eval epoch', 'Metrics', 'Assignment').tic()
            # for sorted_i, i in enumerate(inds):
            #     if tp[sorted_i]:
            #         self.gt_hit_per_prediction.setdefault(i, []).append(pred_gtid_assignment[sorted_i])
            # Timer.get('Eval epoch', 'Metrics', 'Assignment').toc()
            #
            # Timer.get('Eval epoch', 'Metrics', 'Assignment2').tic()
            # inv_ind = np.empty_like(inds)
            # inv_ind[inds] = np.arange(inds.size)
            # tp_inv = tp[inv_ind]
            # gt_hit_per_prediction = np.full_like(tp, fill_value=-1)
            # gt_hit_per_prediction[tp_inv] = pred_gtid_assignment[inv_ind][tp_inv]
            # self.gt_hit_per_prediction2.append(gt_hit_per_prediction)
            # Timer.get('Eval epoch', 'Metrics', 'Assignment2').toc()

        fp = 1 - tp
        assert np.all(fp[pred_gtid_assignment < 0] == 1)

        # compute precision/recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / num_hoi_gt_positives
        prec = tp / (fp + tp)

        # compute average precision
        num_bins = 10  # uniformly distributed in [0, 1) (e.g., use 10 for 0.1 spacing)
        thr_values, thr_inds = np.unique(np.floor(rec * num_bins) / num_bins, return_index=True)
        rec_thresholds = np.full(num_bins + 1, fill_value=-1, dtype=np.int)
        rec_thresholds[np.floor(thr_values * num_bins).astype(np.int)] = thr_inds
        for i in range(num_bins, 0, -1):  # fix gaps of -1s
            if rec_thresholds[i - 1] < 0 and rec_thresholds[i] >= 0:
                rec_thresholds[i - 1] = rec_thresholds[i]
        assert rec_thresholds[0] == 0

        max_p = np.maximum.accumulate(prec[::-1])[::-1]
        ap = np.sum(max_p[rec_thresholds[rec_thresholds >= 0]] / rec_thresholds.size)
        return rec, prec, ap
