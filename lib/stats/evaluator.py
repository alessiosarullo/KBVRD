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
        self.hoi_obj_labels = []
        self.hoi_labels = []
        self.hoi_predictions = []
        self.hoi_gt_pred_assignment = []

        self.metric_functions = {'u-mAP': lambda labels, predictions: average_precision_score(labels, predictions, average='micro'),
                                 'M-mAP': lambda labels, predictions: average_precision_score(labels, predictions, average=None),
                                 'u-rec': lambda labels, predictions: recall_score(np.argmax(labels, axis=1),
                                                                                   np.argmax(predictions, axis=1),
                                                                                   average='micro'),
                                 'M-rec': lambda labels, predictions: recall_score(np.argmax(labels, axis=1),
                                                                                   np.argmax(predictions, axis=1),
                                                                                   average=None),
                                 'u-prc': lambda labels, predictions: precision_score(np.argmax(labels, axis=1),
                                                                                      np.argmax(predictions, axis=1),
                                                                                      average='micro'),
                                 'M-prc': lambda labels, predictions: precision_score(np.argmax(labels, axis=1),
                                                                                      np.argmax(predictions, axis=1),
                                                                                      average=None),
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
        labels = np.concatenate(self.hoi_labels, axis=0)
        predictions = np.concatenate(self.hoi_predictions, axis=0)
        for metric, func in self.metric_functions.items():
            self.metrics[metric] = func(labels, predictions)

    def print_metrics(self, sort=False):
        def _f(_x, _p):
            if _x < 1:
                if _x > 0:
                    return ('%{}.{}f%%'.format(_p + 3, _p)) % (_x * 100)
                else:
                    return ('%{}.{}f%%'.format(_p + 3, 0)) % (_x * 100)
            else:
                return ('%{}d%%'.format(_p + 3)) % 100

        gt_hois = self.dataset.hois
        gt_hoi_hist = Counter(gt_hois[:, 1])
        num_gt_hois = sum(gt_hoi_hist.values())
        if sort:
            inds = [p for p, num in gt_hoi_hist.most_common()]
        else:
            inds = range(self.dataset.num_predicates)

        lines = []
        for k, v in self.metrics.items():
            per_class_str = ' @ [%s]' % ' '.join([_f(x, _p=2) for x in v[inds]]) if v.size > 1 else ''
            lines += ['%7s: %s%s' % (k, _f(np.mean(v), _p=2), per_class_str)]
        lines += ['%8s %8s [%s]' % ('GT HOIs:', 'IDs', ' '.join(['%5d ' % i for i in inds]))]
        lines += ['%8s %8s [%s]' % ('', '%', ' '.join([_f(gt_hoi_hist[i] / num_gt_hois, _p=2) for i in inds]))]
        printstr = '\n'.join(lines)
        print(printstr)
        return printstr

    def process_prediction(self, gt_entry: Example, prediction: Prediction, default='zeros'):
        # TODO docs

        if isinstance(gt_entry, Example):
            gt_hois = gt_entry.gt_hois[:, [0, 2, 1]]  # (h, o, i)
            gt_boxes = gt_entry.gt_boxes.astype(np.float, copy=False)
        else:
            raise ValueError('Unknown type for GT entry: %s.' % str(type(gt_entry)))

        # Ground truth
        num_gt_objs = gt_boxes.shape[0]
        gt_hoi_mat = np.zeros([num_gt_objs, num_gt_objs, self.dataset.num_predicates])
        for h, o, i in gt_hois:
            gt_hoi_mat[h, o, i] = 1

        num_gt_hois = gt_hois.shape[0]
        hoi_labels = np.empty((num_gt_hois, self.dataset.num_predicates))
        hoi_obj_labels = np.empty(num_gt_hois, dtype=np.int)
        for i, (gh, go) in enumerate(gt_hois[:, :2]):
            hoi_labels[i, :] = gt_hoi_mat[gh, go, :]
            hoi_obj_labels[i] = gt_entry.gt_obj_classes[go]
        assert np.all(np.any(hoi_labels, axis=1))

        # Predictions
        hoi_gt_pred_assignment = np.full(num_gt_hois, fill_value=-1, dtype=np.int)
        # For unmatched pairs assume by default:
        if default == 'zeros':  # All zeros
            hoi_predictions = np.zeros((num_gt_hois, self.dataset.num_predicates))
        elif default == 'fg':  # Uniform over foreground interactions
            hoi_predictions = np.ones((num_gt_hois, self.dataset.num_predicates))
            hoi_predictions /= (hoi_predictions.shape[1] - 1)
            hoi_predictions[:, 0] = 0
        elif default == 'bg':  # Delta at null interaction
            hoi_predictions = np.zeros((num_gt_hois, self.dataset.num_predicates))
            hoi_predictions[:, 0] = 1
        else:
            raise ValueError('Unknown value:', default)

        if prediction.ho_pairs is not None:
            assert all([v is not None for v in vars(prediction).values()])
            assert prediction.obj_im_inds.shape[0] == prediction.obj_boxes.shape[0] == prediction.obj_scores.shape[0]
            assert prediction.hoi_img_inds.shape[0] == prediction.ho_pairs.shape[0] == prediction.hoi_score_distributions.shape[0]
            assert len(np.unique(prediction.obj_im_inds)) == len(np.unique(prediction.hoi_img_inds)) == 1

            predict_boxes = prediction.obj_boxes
            predict_ho_pairs = prediction.ho_pairs
            predict_hoi_scores = prediction.hoi_score_distributions
            num_predict_objs = predict_boxes.shape[0]

            predict_hoi_idx_mat = np.full([num_predict_objs, num_predict_objs], fill_value=-1, dtype=np.int)
            for pair_idx, (h, o) in enumerate(predict_ho_pairs):
                predict_hoi_idx_mat[h, o] = pair_idx

            assert num_gt_hois > 0
            gt_pred_ious = compute_ious(gt_boxes, predict_boxes)
            gt_to_predict_box_match = np.argmax(gt_pred_ious, axis=1)
            gt_to_predict_box_match[~gt_pred_ious.any(axis=1)] = -1

            for i, (gh, go) in enumerate(gt_hois[:, :2]):
                ph, po = gt_to_predict_box_match[[gh, go]]
                if ph != -1 and po != -1:
                    idx = predict_hoi_idx_mat[ph, po]
                    if idx >= 0:
                        tmp_scores = predict_hoi_scores[idx, :]
                        assert np.any(tmp_scores)
                        hoi_predictions[i, :] = tmp_scores
                        hoi_gt_pred_assignment[i] = idx
        else:
            assert prediction.hoi_img_inds is None and prediction.ho_pairs is None and prediction.hoi_score_distributions is None

        self.hoi_obj_labels.append(hoi_obj_labels)
        self.hoi_labels.append(hoi_labels)
        self.hoi_predictions.append(hoi_predictions)
        self.hoi_gt_pred_assignment.append(hoi_gt_pred_assignment)
