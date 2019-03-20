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

    def print_metrics(self):
        def _f(_x, _p):
            if _x < 1:
                if _x > 0:
                    return ('%{}.{}f%%'.format(_p + 3, _p)) % (_x * 100)
                else:
                    return ('%{}.{}f%%'.format(_p + 3, 0)) % (_x * 100)
            else:
                return ('%{}d%%'.format(_p + 3)) % 100

        for k, v in self.metrics.items():
            per_class_str = ' @ [%s]' % ' '.join([_f(x, _p=2) for x in v]) if v.size > 1 else ''
            print('%7s: %s%s' % (k, _f(np.mean(v), _p=2), per_class_str))
        gt_hois = self.dataset.hois
        gt_hoi_hist = Counter(gt_hois[:, 1])
        num_gt_hois = sum(gt_hoi_hist.values())
        print('%8s %8s [%s]' % ('GT HOIs:', 'IDs', ' '.join(['%5d ' % i for i in range(self.dataset.num_predicates)])))
        print('%8s %8s [%s]' % ('', '%', ' '.join([_f(gt_hoi_hist[i] / num_gt_hois, _p=2) for i in range(self.dataset.num_predicates)])))

    def process_prediction(self, gt_entry: Example, prediction: Prediction):
        # TODO docs

        if isinstance(gt_entry, Example):
            gt_hois = gt_entry.gt_hois[:, [0, 2, 1]]  # (h, o, i)
            gt_boxes = gt_entry.gt_boxes.astype(np.float, copy=False)
        else:
            raise ValueError('Unknown type for GT entry: %s.' % str(type(gt_entry)))

        if not prediction.is_complete():
            return None, None
        assert len(np.unique(prediction.obj_im_inds)) == len(np.unique(prediction.hoi_img_inds)) == 1

        predict_boxes = prediction.obj_boxes
        predict_ho_pairs = prediction.ho_pairs
        predict_hoi_scores = prediction.hoi_score_distributions

        num_gt_objs = gt_boxes.shape[0]
        num_predict_objs = predict_boxes.shape[0]

        gt_hoi_mat = np.zeros([num_gt_objs, num_gt_objs, self.dataset.num_predicates])
        for h, o, i in gt_hois:
            gt_hoi_mat[h, o, i] = 1

        predict_hoi_mat = np.zeros([num_predict_objs, num_predict_objs, predict_hoi_scores.shape[1]])
        for pair_idx, (h, o) in enumerate(predict_ho_pairs):
            predict_hoi_mat[h, o, :] = predict_hoi_scores[pair_idx, :]

        assert gt_hois.shape[0] > 0
        gt_pred_ious = compute_ious(gt_boxes, predict_boxes)
        gt_to_predict_box_match = np.argmax(gt_pred_ious, axis=1)
        gt_to_predict_box_match[~gt_pred_ious.any(axis=1)] = -1

        hoi_labels = np.empty((gt_hois.shape[0], self.dataset.num_predicates), dtype=gt_hoi_mat.dtype)

        # For unmatched pairs assume by default either all 0s, a uniform distribution over foreground interactions or a delta at null interaction.
        # All zeros
        hoi_predictions = np.zeros((gt_hois.shape[0], self.dataset.num_predicates), dtype=predict_hoi_mat.dtype)
        # Uniform over foreground
        # hoi_predictions = np.ones((gt_hois.shape[0], self.dataset.num_predicates), dtype=predict_hoi_mat.dtype)
        # hoi_predictions /= (hoi_predictions.shape[1] - 1)
        # hoi_predictions[:, 0] = 0
        # Delta
        # hoi_predictions = np.zeros((gt_hois.shape[0], self.dataset.num_predicates), dtype=predict_hoi_mat.dtype)
        # hoi_predictions[:, 0] = 1
        for i, (gh, go) in enumerate(gt_hois[:, :2]):
            hoi_labels[i, :] = gt_hoi_mat[gh, go, :]

            ph, po = gt_to_predict_box_match[[gh, go]]
            if ph != -1 and po != -1 and np.any(predict_hoi_mat[ph, po, :]):
                hoi_predictions[i, :] = predict_hoi_mat[ph, po, :]
        assert np.all(np.any(hoi_labels, axis=1))

        self.hoi_labels.append(hoi_labels)
        self.hoi_predictions.append(hoi_predictions)
