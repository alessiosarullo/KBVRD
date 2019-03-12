from typing import List, Dict

import numpy as np

from lib.bbox_utils import compute_ious
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.dataset.utils import Example
from lib.models.utils import Prediction

from sklearn.metrics import average_precision_score


class EvalStats:
    def __init__(self, dataset: HicoDetInstanceSplit, iou_thresh=0.5):
        self.iou_thresh = iou_thresh
        self.dataset = dataset
        self.hoi_labels = []
        self.hoi_predictions = []

    @classmethod
    def evaluate_predictions(cls, dataset: HicoDetInstanceSplit, predictions: List[Dict], **kwargs):
        assert len(predictions) == dataset.num_images, (len(predictions), dataset.num_images)

        eval_stats = cls(dataset, **kwargs)
        for i, res in enumerate(predictions):
            ex = dataset.get_entry(i, read_img=False, ignore_precomputed=True)
            prediction = Prediction.from_dict(res)
            eval_stats.process_prediction(ex, prediction)

        return eval_stats  # type: EvalStats

    def print(self):
        def _f(_x, _p):
            if _x < 1:
                if _x > 0:
                    return ('%{}.{}f%%'.format(_p + 3, _p)) % (_x * 100)
                else:
                    return ('%{}.{}f%%'.format(_p + 3, 0)) % (_x * 100)
            else:
                return ('%{}d%%'.format(_p + 3)) % 100

        labels = np.concatenate(self.hoi_labels, axis=0)
        predictions = np.concatenate(self.hoi_predictions, axis=0)
        micro_map = average_precision_score(labels, predictions, average='micro')
        pc_map = average_precision_score(labels, predictions, average=None)
        print('  umAP: %s' % _f(np.mean(micro_map), _p=2))
        print('pc-mAP: %s @ [%s]' % (_f(np.mean(pc_map), _p=2), ' '.join([_f(pcap, _p=2) for pcap in pc_map])))

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

        hoi_labels, hoi_predictions = [], []
        for gh, go in gt_hois[:, :2]:
            hoi_labels.append(gt_hoi_mat[gh, go, :])

            ph, po = gt_to_predict_box_match[[gh, go]]
            if ph != -1 and po != -1:
                hoi_predictions.append(predict_hoi_mat[ph, po, :])
            else:
                hoi_predictions.append(np.zeros(predict_hoi_mat.shape[2]))

        self.hoi_labels.append(np.stack(hoi_labels, axis=0))
        self.hoi_predictions.append(np.stack(hoi_predictions, axis=0))
