from collections import Counter
from typing import List, Dict

import numpy as np

from lib.bbox_utils import compute_ious
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.dataset.utils import Example
from lib.models.utils import Prediction


class Evaluator:
    def __init__(self, dataset: HicoDetInstanceSplit, iou_thresh=0.5):
        self.iou_thresh = iou_thresh
        self.dataset = dataset
        self.op_pair_to_inter, self.inter_to_op_pair = self.parse_interactions()

        self.gt_im_inds = []
        self.gt_hois = []
        self.gt_box_pairs = []
        self.predict_im_inds = []
        self.predict_hoi_scores = []
        self.predict_box_pairs = []

        self.metrics = {}  # type: Dict[str, np.ndarray]

    def parse_interactions(self):
        num_interactions = len(self.dataset.hicodet.interaction_list)
        op_pair_to_inter = np.full([self.dataset.num_object_classes, self.dataset.num_predicates], fill_value=-1, dtype=np.int)
        inter_to_op_pair = np.empty([num_interactions, 2], dtype=np.int)
        for iid in range(num_interactions):
            obj_id = self.dataset.hicodet.get_object_index(iid)
            pred_id = self.dataset.hicodet.get_predicate_index(iid)
            op_pair_to_inter[obj_id, pred_id] = iid
            inter_to_op_pair[iid, :] = [obj_id, pred_id]
        assert np.sum(op_pair_to_inter >= 0) == 600, np.sum(op_pair_to_inter >= 0)
        return op_pair_to_inter, inter_to_op_pair

    @property
    def num_interactions(self):
        return self.inter_to_op_pair.shape[0]

    @classmethod
    def evaluate_predictions(cls, dataset: HicoDetInstanceSplit, predictions: List[Dict], **kwargs):
        assert len(predictions) == dataset.num_images, (len(predictions), dataset.num_images)

        evaluator = cls(dataset, **kwargs)
        for i, res in enumerate(predictions):
            ex = dataset.get_entry(i, read_img=False, ignore_precomputed=True)
            prediction = Prediction.from_dict(res)
            evaluator.process_prediction(i, ex, prediction)

        evaluator.compute_metrics()
        return evaluator  # type: Evaluator

    def compute_metrics(self):
        gt_im_inds = np.concatenate(self.gt_im_inds, axis=0)
        gt_ho_boxes = np.concatenate(self.gt_box_pairs, axis=0)
        gt_hoi_inds = np.concatenate(self.gt_hois, axis=0)
        predict_im_inds = np.concatenate(self.predict_im_inds, axis=0)
        predict_ho_boxes = np.concatenate(self.predict_box_pairs, axis=0)
        predict_hoi_scores = np.concatenate(self.predict_hoi_scores, axis=0)

        ap = np.zeros(self.num_interactions)
        recall = np.zeros(self.num_interactions)
        for j in range(self.num_interactions):
            gt_inter_inds = (gt_hoi_inds == j)
            # FIXME this uses all pairs for every interaction, which will drive precision down. Maybe threshold by score?
            rec_j, prec_j, ap_j = self.eval_interactions(predict_im_inds, predict_ho_boxes, predict_hoi_scores[:, j],
                                                         gt_im_inds[gt_inter_inds], gt_ho_boxes[gt_inter_inds, :])
            ap[j] = ap_j
            if rec_j.size > 0:
                recall[j] = rec_j[-1]

        self.metrics['M-mAP'] = ap
        self.metrics['M-rec'] = recall

    def print_metrics(self, sort=False):
        mf = MetricFormatter()
        lines = []

        obj_metrics = {k: v for k, v in self.metrics.items() if k.lower().startswith('obj')}
        lines += mf.format_metric_and_gt_lines(self.dataset.obj_labels, metrics=obj_metrics, gt_str='GT objects', sort=sort)

        hoi_metrics = {k: v for k, v in self.metrics.items() if not k.lower().startswith('obj')}
        lines += mf.format_metric_and_gt_lines(self.dataset.hois[:, 1], metrics=hoi_metrics, gt_str='GT HOIs', sort=sort)

        printstr = '\n'.join(lines)
        print(printstr)
        return printstr

    def process_prediction(self, im_id, gt_entry: Example, prediction: Prediction):
        if isinstance(gt_entry, Example):
            gt_hois = gt_entry.gt_hois[:, [0, 2, 1]]  # (h, o, i)
            gt_boxes = gt_entry.gt_boxes.astype(np.float, copy=False)
            gt_obj_classes = gt_entry.gt_obj_classes
            gt_ho_boxes = np.concatenate([gt_boxes[gt_hois[:, 0], :], gt_boxes[gt_hois[:, 1], :]], axis=1)
            gt_hoi_inds = self.op_pair_to_inter[gt_obj_classes[gt_hois[:, 1]], gt_hois[:, 2]]
            assert np.all(gt_hoi_inds) >= 0
        else:
            raise ValueError('Unknown type for GT entry: %s.' % str(type(gt_entry)))

        predict_ho_boxes = np.zeros([0, 8])
        predict_hoi_scores = np.zeros([0, self.inter_to_op_pair.shape[0]])
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
                predict_ho_boxes = np.concatenate([predict_boxes[predict_ho_pairs[:, 0], :], predict_boxes[predict_ho_pairs[:, 1], :]], axis=1)
                try:
                    predict_hoi_scores = prediction.hoi_scores
                except AttributeError:
                    predict_action_scores = prediction.action_score_distributions
                    predict_obj_scores_per_ho_pair = predict_obj_scores[predict_ho_pairs[:, 1], :]

                    predict_hoi_scores = np.empty([predict_ho_pairs.shape[0], self.inter_to_op_pair.shape[0]])
                    for iid, (oid, pid) in enumerate(self.inter_to_op_pair):
                        predict_hoi_scores[:, iid] = predict_obj_scores_per_ho_pair[:, oid] * predict_action_scores[:, pid]

            else:
                assert prediction.hoi_img_inds is None and prediction.ho_pairs is None and prediction.action_score_distributions is None
        else:
            assert prediction.ho_pairs is None

        self.gt_im_inds.append(np.full(gt_hoi_inds.shape[0], fill_value=im_id))
        self.gt_hois.append(gt_hoi_inds)
        self.gt_box_pairs.append(gt_ho_boxes)
        self.predict_im_inds.append(np.full(predict_hoi_scores.shape[0], fill_value=im_id))
        self.predict_hoi_scores.append(predict_hoi_scores)
        self.predict_box_pairs.append(predict_ho_boxes)

    def eval_interactions(self, predicted_im_ids, predicted_bb_pairs, predicted_conf_scores, gt_im_ids, gt_bb_pairs, min_overlap=0.5):
        """
        Equivalent to VOCevaldet_bboxpair in the original MATLAB code.
        """
        num_pos = gt_bb_pairs.shape[0]
        gt_det = np.zeros(num_pos, dtype=bool)
        inds = np.argsort(predicted_conf_scores)[::-1]
        predicted_bb_pairs = predicted_bb_pairs[inds]
        predicted_im_ids = predicted_im_ids[inds]

        num_predictions = predicted_bb_pairs.shape[0]
        tp = np.zeros(num_predictions)
        fp = np.zeros(num_predictions)

        for d in range(num_predictions):
            im_id = predicted_im_ids[d]
            bb_1 = predicted_bb_pairs[d, :4]
            bb_2 = predicted_bb_pairs[d, 4:]

            ov_max = -np.inf
            j_max = None
            for j in range(gt_bb_pairs.shape[0]):
                if gt_im_ids[j] == im_id:
                    bbgt_1 = gt_bb_pairs[j, :4]
                    bbgt_2 = gt_bb_pairs[j, 4:]
                    ov_1 = compute_ious(bb_1[None, :], bbgt_1[None, :]).squeeze()
                    ov_2 = compute_ious(bb_2[None, :], bbgt_2[None, :]).squeeze()
                    min_ov = min(ov_1, ov_2)
                    if min_ov > ov_max:
                        ov_max = min_ov
                        j_max = j
            if ov_max >= min_overlap:
                assert j_max is not None
                if not gt_det[j_max]:
                    tp[d] = 1
                    gt_det[j_max] = True
                else:
                    fp[d] = 1  # false positive (multiple detection)
            else:
                fp[d] = 1

        # compute precision/recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / num_pos
        prec = tp / (fp + tp)

        # compute average precision
        ap = 0
        for t in np.arange(11) / 10:
            pr = prec[rec >= t]
            if pr.size > 0:
                p = max(pr)
            else:
                p = 0
            ap = ap + p / 11
        return rec, prec, ap


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
