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
        self.dataset = dataset
        self.hoi_labels = []
        self.hoi_predictions = []
        self.unmatched_gt_hois = []

        self.hoi_metric_functions = {'u-mAP': lambda labels, predictions: average_precision_score(labels, predictions, average='micro'),
                                     'M-mAP': lambda labels, predictions: average_precision_score(labels, predictions, average=None),
                                     'u-rec': lambda labels, predictions: recall_score(labels, (predictions > 0.5), average='micro'),
                                     'M-rec': lambda labels, predictions: recall_score(labels, (predictions > 0.5), average=None),
                                     'u-prc': lambda labels, predictions: precision_score(labels, (predictions > 0.5), average='micro'),
                                     'M-prc': lambda labels, predictions: precision_score(labels, (predictions > 0.5), average=None),
                                     }
        self.metrics = {}  # type: Dict[str, np.ndarray]

    @classmethod
    def evaluate_predictions(cls, dataset: HicoDetInstanceSplit, predictions: List[Dict], **kwargs):
        assert len(predictions) == dataset.num_images, (len(predictions), dataset.num_images)

        evaluator = cls(dataset, **kwargs)
        for i, res in enumerate(predictions):
            ex = dataset.get_entry(i, read_img=False, ignore_precomputed=True)
            prediction = Prediction.from_dict(res)
            evaluator.process_prediction(ex, prediction)

        evaluator.compute_metrics()
        return evaluator  # type: Evaluator

    def compute_metrics(self):
        hoi_labels = np.concatenate(self.hoi_labels, axis=0)
        hoi_predictions = np.concatenate(self.hoi_predictions, axis=0)
        for metric, func in self.hoi_metric_functions.items():
            self.metrics[metric] = func(hoi_labels, hoi_predictions)

    def print_metrics(self, sort=False):
        mf = MetricFormatter()
        lines = []

        obj_metrics = {k: v for k, v in self.metrics.items() if k.lower().startswith('obj')}
        lines += mf.format_metric_and_gt_lines(self.dataset.obj_labels, metrics=obj_metrics, gt_str='GT objects', sort=sort)

        hoi_metrics = {k: v for k, v in self.metrics.items() if not k.lower().startswith('obj')}
        lines += mf.format_metric_and_gt_lines(self.dataset.hois[:, 1], metrics=hoi_metrics, gt_str='GT HOIs', sort=sort)

        all_predictions = np.concatenate(self.hoi_predictions, axis=0)
        all_labels = np.concatenate(self.hoi_labels, axis=0)
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

        # Ground truth
        num_gt_objs = gt_boxes.shape[0]
        obj_labels = np.zeros([num_gt_objs, self.dataset.num_object_classes])
        obj_labels[np.arange(obj_labels.shape[0]), gt_obj_classes] = 1

        predict_ho_pairs = np.zeros((0, 2), dtype=np.int)
        predict_boxes = np.zeros((0, 4))
        predict_obj_scores = np.zeros((0, self.dataset.num_object_classes))
        predict_hoi_scores = np.zeros((0, self.dataset.num_predicates))
        if prediction.obj_boxes is not None:
            assert prediction.obj_im_inds.shape[0] == prediction.obj_boxes.shape[0] == prediction.obj_scores.shape[0]
            assert prediction.obj_im_inds is not None and prediction.obj_boxes is not None and prediction.obj_scores is not None

            predict_obj_scores = prediction.obj_scores
            predict_boxes = prediction.obj_boxes

            if prediction.ho_pairs is not None:
                assert all([v is not None for v in vars(prediction).values()])
                assert prediction.ho_img_inds.shape[0] == prediction.ho_pairs.shape[0] == prediction.action_score_distributions.shape[0]
                assert len(np.unique(prediction.obj_im_inds)) == len(np.unique(prediction.ho_img_inds)) == 1

                predict_ho_pairs = prediction.ho_pairs
                predict_hoi_scores = prediction.action_score_distributions

                # fg_hois = predict_hoi_scores[:, 0] < 0.5  # FIXME magic constant
                # predict_hoi_scores = predict_hoi_scores[fg_hois, :]
                # predict_ho_pairs = predict_ho_pairs[fg_hois, :]
            else:
                assert prediction.ho_img_inds is None and prediction.ho_pairs is None and prediction.action_score_distributions is None
        else:
            assert prediction.ho_pairs is None

        num_gt_hois = gt_hois.shape[0]
        num_predictions = predict_hoi_scores.shape[0]
        pred_gt_ious = compute_ious(predict_boxes, gt_boxes)
        pred_gt_ious_class_match = np.argmax(predict_obj_scores, axis=1)[:, None] == gt_obj_classes[None, :]

        unmatched_gt_hois = np.ones(num_gt_hois, dtype=bool)
        hoi_predictions = predict_hoi_scores.copy()
        hoi_labels = np.zeros((num_predictions, self.dataset.num_predicates))
        for predict_idx, (ph, po) in enumerate(predict_ho_pairs):
            gt_pair_ious = np.zeros(num_gt_hois)
            for gtidx, (gh, go, gi) in enumerate(gt_hois):
                iou_h = pred_gt_ious[ph, gh]
                iou_o = pred_gt_ious[po, go]
                if pred_gt_ious_class_match[ph, gh] and pred_gt_ious_class_match[po, go]:
                    gt_pair_ious[gtidx] = min(iou_h, iou_o)
            if np.any(gt_pair_ious > self.iou_thresh):
                gtidxs = (gt_pair_ious > self.iou_thresh)
                hoi_labels[predict_idx, np.unique(gt_hois[gtidxs, 2])] = 1
                unmatched_gt_hois[gtidxs] = False

        num_unmatched_gt_hois = np.sum(unmatched_gt_hois)
        hoi_unmatched_labels = np.zeros((num_unmatched_gt_hois, self.dataset.num_predicates))
        hoi_unmatched_labels[np.arange(num_unmatched_gt_hois), gt_hois[unmatched_gt_hois, 2]] = 1

        # Add unassigned predictions
        hoi_labels = np.concatenate([hoi_labels, hoi_unmatched_labels], axis=0)
        hoi_predictions = np.concatenate([hoi_predictions, np.zeros((num_unmatched_gt_hois, self.dataset.num_predicates))], axis=0)

        self.hoi_labels.append(hoi_labels)
        self.hoi_predictions.append(hoi_predictions)
        self.unmatched_gt_hois.append(unmatched_gt_hois)


class MetricFormatter:
    def __init__(self):
        super().__init__()

    def format_metric_and_gt_lines(self, gt_labels, metrics, gt_str, sort=False):
        lines = []
        gt_label_hist = Counter(gt_labels)
        num_gt_examples = sum(gt_label_hist.values())
        if sort:
            inds = [p for p, num in gt_label_hist.most_common()]
        else:
            inds = range(len(gt_label_hist))

        for k, v in metrics.items():
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
