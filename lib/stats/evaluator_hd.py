from collections import Counter
from typing import List, Dict

import numpy as np
from sklearn.metrics import average_precision_score, recall_score, precision_score

from lib.bbox_utils import compute_ious
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.dataset.utils import Example
from lib.models.utils import Prediction


class Evaluator:
    def __init__(self, dataset: HicoDetInstanceSplit, iou_thresh=0.5):
        self.iou_thresh = iou_thresh
        self.filter_bg = False
        self.filter_unknown_pairs = True

        self.dataset = dataset
        self.hoi_obj_labels = []
        self.hoi_labels = []
        self.hoi_predictions = []
        self.hoi_gt_pred_assignment = []
        self.num_unmatched_gt_hois = 0

        self.hoi_metric_functions = {
            'M-mAP': lambda labels, predictions: average_precision_score(labels, predictions, average=None),
            'M-rec': lambda labels, predictions: recall_score(labels, (predictions > 0.5)),
            'M-prc': lambda labels, predictions: precision_score(labels, (predictions > 0.5)),
        }
        self.metrics = {}  # type: Dict[str, np.ndarray]

        self.known_pairs, self.num_interactions = self.get_known_pairs()

    def get_known_pairs(self):
        known_pairs = np.zeros((self.dataset.num_object_classes, self.dataset.num_predicates), dtype=bool)
        for iid in range(len(self.dataset.hicodet.interactions)):
            obj_id = self.dataset.hicodet.get_object_index(iid)
            pred_id = self.dataset.hicodet.get_predicate_index(iid)
            known_pairs[obj_id, pred_id] = 1
        assert np.sum(known_pairs) == 600, np.sum(known_pairs)

        num_occurrences = np.zeros((self.dataset.num_object_classes, self.dataset.num_predicates), dtype=np.int)
        for h, i, o in self.dataset.hois:
            num_occurrences[o, i] += 1
        assert np.sum(num_occurrences > 0) == 600

        known_pairs = np.flatnonzero(known_pairs.reshape(-1))
        num_interactions = num_occurrences.reshape(-1)[known_pairs]
        assert num_interactions.sum() == num_occurrences.sum()
        return known_pairs, num_interactions

    @classmethod
    def evaluate_predictions(cls, dataset: HicoDetInstanceSplit, predictions: List[Dict], **kwargs):
        assert len(predictions) == dataset.num_images, (len(predictions), dataset.num_images)

        evaluator = cls(dataset, **kwargs)
        for i, res in enumerate(predictions):
            ex = dataset.get_entry(i, read_img=False, ignore_precomputed=True)
            prediction = Prediction.from_dict(res)
            evaluator.process_prediction(ex, prediction)

        evaluator.hoi_labels = np.concatenate(evaluator.hoi_labels, axis=0)
        evaluator.hoi_predictions = np.concatenate(evaluator.hoi_predictions, axis=0)
        evaluator.compute_metrics()
        return evaluator  # type: Evaluator

    def compute_metrics(self):
        for metric_name, func in self.hoi_metric_functions.items():
            metric = np.zeros(self.known_pairs.size)
            for j in range(self.known_pairs.size):
                metric[j] = func(self.hoi_labels[:, j], self.hoi_predictions[:, j])
            self.metrics[metric_name] = metric

    def print_metrics(self, sort=False):
        mf = MetricFormatter()
        lines = []

        obj_metrics = {k: v for k, v in self.metrics.items() if k.lower().startswith('obj')}
        lines += mf.format_metric_and_gt_lines(gt_label_hist=Counter(self.dataset.obj_labels), metrics=obj_metrics, gt_str='GT objects', sort=sort)

        hoi_metrics = {k: v for k, v in self.metrics.items() if not k.lower().startswith('obj')}
        lines += mf.format_metric_and_gt_lines(gt_label_hist=Counter({i: n for i, n in enumerate(self.num_interactions)}),
                                               metrics=hoi_metrics, gt_str='GT HOIs', sort=sort)

        all_predictions = self.hoi_predictions
        all_labels = self.hoi_labels
        actual_predictions = np.any(all_predictions, axis=1)
        actual_labels = np.any(all_labels, axis=1).sum()
        lines += ['%30s: %6d.' % ('Num predicted HOIs', actual_predictions.sum())]
        lines += ['%30s: %6d.' % ('Num unassigned predictions', np.all(all_labels == 0, axis=1).sum())]
        lines += ['%30s: %6d.' % ('Num GT HOIs', actual_labels.sum())]
        lines += ['%30s: %6d.' % ('Num unmatched GT HOIs', np.all(all_predictions == 0, axis=1).sum())]
        lines += ['%30s: %6d.' % ('Num total', all_predictions.shape[0])]

        printstr = '\n'.join(lines)
        print(printstr)
        return printstr

    def process_prediction(self, gt_entry: Example, prediction: Prediction):
        if isinstance(gt_entry, Example):
            gt_hois = gt_entry.gt_hois[:, [0, 2, 1]]  # (h, o, i)
            gt_boxes = gt_entry.gt_boxes.astype(np.float, copy=False)
            gt_obj_classes = gt_entry.gt_obj_classes
        else:
            raise ValueError('Unknown type for GT entry: %s.' % str(type(gt_entry)))

        num_possible_hois = self.dataset.num_object_classes * self.dataset.num_predicates

        # # Ground truth objects
        # num_gt_objs = gt_boxes.shape[0]
        # obj_labels = np.zeros([num_gt_objs, self.dataset.num_object_classes])
        # obj_labels[np.arange(obj_labels.shape[0]), gt_obj_classes] = 1

        predict_ho_pairs = np.zeros((0, 2), dtype=np.int)
        predict_boxes = np.zeros((0, 4))
        predict_hoi_scores = np.zeros((0, num_possible_hois))
        if prediction.obj_boxes is not None:
            assert prediction.obj_im_inds.shape[0] == prediction.obj_boxes.shape[0] == prediction.obj_scores.shape[0]
            assert prediction.obj_im_inds is not None and prediction.obj_boxes is not None and prediction.obj_scores is not None

            predict_obj_scores = prediction.obj_scores
            predict_boxes = prediction.obj_boxes

            if prediction.ho_pairs is not None:
                assert all([v is not None for v in vars(prediction).values()])
                assert prediction.hoi_img_inds.shape[0] == prediction.ho_pairs.shape[0] == prediction.action_score_distributions.shape[0]
                assert len(np.unique(prediction.obj_im_inds)) == len(np.unique(prediction.hoi_img_inds)) == 1

                predict_ho_pairs = prediction.ho_pairs
                predict_action_scores = prediction.action_score_distributions
                predict_hoi_obj_scores = predict_obj_scores[predict_ho_pairs[:, 1], :]
                predict_hoi_scores = (predict_hoi_obj_scores[:, :, None] * predict_action_scores[:, None, :]).reshape(predict_ho_pairs.shape[0], -1)
                assert predict_hoi_scores.shape[1] == num_possible_hois

                # fg_hois = predict_action_scores[:, 0] < 0.5  # FIXME magic constant
                # predict_action_scores = predict_action_scores[fg_hois, :]
                # predict_ho_pairs = predict_ho_pairs[fg_hois, :]
            else:
                assert prediction.hoi_img_inds is None and prediction.ho_pairs is None and prediction.action_score_distributions is None
        else:
            assert prediction.ho_pairs is None

        num_gt_hois = gt_hois.shape[0]
        num_predictions = predict_hoi_scores.shape[0]
        pred_gt_ious = compute_ious(predict_boxes, gt_boxes)

        unmatched_gt_hois = np.ones(num_gt_hois, dtype=bool)
        hoi_predictions = predict_hoi_scores
        hoi_labels = np.zeros((num_predictions, num_possible_hois))
        for predict_idx, (ph, po) in enumerate(predict_ho_pairs):
            gt_pair_ious = np.zeros(num_gt_hois)
            for gtidx, (gh, go, gi) in enumerate(gt_hois):
                iou_h = pred_gt_ious[ph, gh]
                iou_o = pred_gt_ious[po, go]
                gt_pair_ious[gtidx] = min(iou_h, iou_o)
            if np.any(gt_pair_ious > self.iou_thresh):
                gtidxs = (gt_pair_ious > self.iou_thresh)
                gt_op_hoi_labels = np.zeros((self.dataset.num_object_classes, self.dataset.num_predicates))
                gt_op_hoi_labels[gt_obj_classes[gt_hois[gtidxs, 1]], gt_hois[gtidxs, 2]] = 1
                hoi_labels[predict_idx, :] = gt_op_hoi_labels.reshape(-1)
                unmatched_gt_hois[gtidxs] = False

        num_unmatched_gt_hois = np.sum(unmatched_gt_hois)
        if num_unmatched_gt_hois > 0:
            hoi_unmatched_labels = np.zeros((num_unmatched_gt_hois, self.dataset.num_object_classes, self.dataset.num_predicates))
            hoi_unmatched_labels[np.arange(num_unmatched_gt_hois), gt_obj_classes[gt_hois[unmatched_gt_hois, 1]], gt_hois[unmatched_gt_hois, 2]] = 1
            hoi_unmatched_labels = hoi_unmatched_labels.reshape(hoi_unmatched_labels.shape[0], -1)
        else:
            hoi_unmatched_labels = np.zeros((num_unmatched_gt_hois, num_possible_hois))

        # Add unassigned predictions
        hoi_labels = np.concatenate([hoi_labels, hoi_unmatched_labels], axis=0)
        hoi_predictions = np.concatenate([hoi_predictions, np.zeros((num_unmatched_gt_hois, hoi_predictions.shape[1]))], axis=0)

        if self.filter_unknown_pairs:
            hoi_labels = hoi_labels[:, self.known_pairs]
            hoi_predictions = hoi_predictions[:, self.known_pairs]
        self.hoi_labels.append(hoi_labels.astype(np.float32))
        self.hoi_predictions.append(hoi_predictions.astype(np.float32))


class MetricFormatter:
    def __init__(self):
        super().__init__()

    def format_metric_and_gt_lines(self, gt_label_hist, metrics, gt_str, sort=False):
        lines = []
        num_gt_examples = sum(gt_label_hist.values())
        if sort:
            inds = [p for p, num in gt_label_hist.most_common()]
        else:
            inds = range(len(gt_label_hist))

        for k, v in metrics.items():
            assert (len(inds) == v.size) or v.size == 1
            lines += [self.format_metric(k, v[inds] if v.size > 1 else v, len(gt_str))]
        format_str = '%{}s %8s [%s]'.format(len(gt_str) + 1)
        lines += [format_str % ('%s:' % gt_str, 'IDs', ' '.join(['%5d ' % i for i in inds]))]
        lines += [format_str % ('', '%', ' '.join([self._format_percentage(gt_label_hist[i] / num_gt_examples) for i in inds]))]
        return lines

    def format_metric(self, metric_name, data, metric_str_len=None):
        metric_str_len = metric_str_len or len(metric_name)
        per_class_str = ' @ [%s]' % ' '.join([self._format_percentage(x) for x in data]) if data.size > 1 else ''
        f_str = ('%{}s: %s%s'.format(metric_str_len)) % (metric_name, self._format_percentage(np.mean(data)), per_class_str)
        return f_str

    @staticmethod
    def _format_percentage(value, precision=2):
        if value < 1:
            if value > 0:
                return ('%{}.{}f%%'.format(precision + 3, precision)) % (value * 100)
            else:
                return ('%{}.{}f%%'.format(precision + 3, 0)) % (value * 100)
        else:
            return ('%{}d%%'.format(precision + 3)) % 100
