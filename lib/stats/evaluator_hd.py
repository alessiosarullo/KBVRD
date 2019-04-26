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

        self.gt_hoi_classes = []
        self.predict_hoi_scores = []
        self.pred_gt_ho_assignment = []

        self.gt_count = 0

        self.metrics = {}  # type: Dict[str, np.ndarray]

    def parse_interactions(self):
        interactions = self.dataset.hicodet.interactions
        num_interactions = interactions.shape[0]

        op_pair_to_inter = np.full([self.dataset.num_object_classes, self.dataset.num_predicates], fill_value=-1, dtype=np.int)
        op_pair_to_inter[interactions[:, 1], interactions[:, 0]] = np.arange(num_interactions)

        inter_to_op_pair = interactions[:, [1, 0]]

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
        print('Prediction processed. Computing metrics.')
        evaluator.compute_metrics()
        return evaluator  # type: Evaluator

    def compute_metrics(self):
        gt_hoi_classes = np.concatenate(self.gt_hoi_classes, axis=0)
        assert self.gt_count == gt_hoi_classes.shape[0]
        predict_hoi_scores = np.concatenate(self.predict_hoi_scores, axis=0)
        pred_gt_ho_assignment = np.concatenate(self.pred_gt_ho_assignment, axis=0)

        gt_hoi_classes_count = Counter(gt_hoi_classes.tolist())

        ap = np.zeros(self.num_interactions)
        recall = np.zeros(self.num_interactions)
        for j in range(self.num_interactions):
            # FIXME this uses all pairs for every interaction, which will drive precision down. Maybe threshold by score?
            rec_j, prec_j, ap_j = self.eval_interactions(predict_hoi_scores[:, j], pred_gt_ho_assignment[:, j], gt_hoi_classes_count[j])
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
            gt_hoi_triplets = gt_entry.gt_hois[:, [0, 2, 1]]  # (h, o, i)
            num_gt_hois = gt_hoi_triplets.shape[0]

            gt_boxes = gt_entry.gt_boxes.astype(np.float, copy=False)

            gt_hoi_classes = self.op_pair_to_inter[gt_entry.gt_obj_classes[gt_hoi_triplets[:, 1]], gt_hoi_triplets[:, 2]]
            assert np.all(gt_hoi_classes) >= 0

            gt_ho_ids = self.gt_count + np.arange(num_gt_hois)
            self.gt_count += num_gt_hois
        else:
            raise ValueError('Unknown type for GT entry: %s.' % str(type(gt_entry)))

        predict_hoi_scores = np.zeros([0, self.inter_to_op_pair.shape[0]])
        predict_ho_pairs = np.zeros((0, 2), dtype=np.int)
        predict_boxes = np.zeros((0, 4))
        if prediction.obj_boxes is not None:
            assert prediction.obj_im_inds.shape[0] == prediction.obj_boxes.shape[0] == prediction.obj_scores.shape[0]
            assert prediction.obj_im_inds is not None and prediction.obj_boxes is not None and prediction.obj_scores is not None

            predict_boxes = prediction.obj_boxes

            if prediction.ho_pairs is not None:
                assert all([v is not None for v in vars(prediction).values()])
                assert len(np.unique(prediction.obj_im_inds)) == len(np.unique(prediction.hoi_img_inds)) == 1

                predict_ho_pairs = prediction.ho_pairs
                try:
                    predict_hoi_scores = prediction.hoi_scores
                except AttributeError:
                    predict_action_scores = prediction.action_score_distributions
                    predict_obj_scores_per_ho_pair = prediction.obj_scores[predict_ho_pairs[:, 1], :]

                    predict_hoi_scores = np.empty([predict_ho_pairs.shape[0], self.inter_to_op_pair.shape[0]])
                    for iid, (oid, pid) in enumerate(self.inter_to_op_pair):
                        predict_hoi_scores[:, iid] = predict_obj_scores_per_ho_pair[:, oid] * predict_action_scores[:, pid]
        else:
            assert prediction.ho_pairs is None

        pred_gt_ious = compute_ious(predict_boxes, gt_boxes)
        pred_gt_assignment_per_hoi = np.full((predict_hoi_scores.shape[0], self.num_interactions), fill_value=-1, dtype=np.int)
        for predict_idx, (ph, po) in enumerate(predict_ho_pairs):
            gt_pair_ious = np.zeros(num_gt_hois)
            for gtidx, (gh, go, gi) in enumerate(gt_hoi_triplets):
                iou_h = pred_gt_ious[ph, gh]
                iou_o = pred_gt_ious[po, go]
                gt_pair_ious[gtidx] = min(iou_h, iou_o)
            if np.any(gt_pair_ious >= self.iou_thresh):
                gt_pair_ious_per_hoi = np.zeros((num_gt_hois, self.num_interactions))
                gt_pair_ious_per_hoi[np.arange(num_gt_hois), gt_hoi_classes] = gt_pair_ious
                gt_assignments = gt_pair_ious_per_hoi.argmax(axis=0)[np.any(gt_pair_ious_per_hoi >= self.iou_thresh, axis=0)]
                gt_hoi_assignments = gt_hoi_classes[gt_assignments]
                assert np.unique(gt_assignments).size == gt_assignments.size
                assert np.unique(gt_hoi_assignments).size == gt_hoi_assignments.size
                pred_gt_assignment_per_hoi[predict_idx, gt_hoi_assignments] = gt_ho_ids[gt_assignments]

        self.gt_hoi_classes.append(gt_hoi_classes)
        self.predict_hoi_scores.append(predict_hoi_scores)
        self.pred_gt_ho_assignment.append(pred_gt_assignment_per_hoi)

    def eval_interactions(self, predicted_conf_scores, pred_gt_ho_assignment, num_hoi_gt_positives):
        """
        Equivalent to VOCevaldet_bboxpair in the original MATLAB code.
        """
        gt_assigned = np.zeros(self.gt_count, dtype=bool)
        inds = np.argsort(predicted_conf_scores)[::-1]
        pred_gt_ho_assignment = pred_gt_ho_assignment[inds]

        num_predictions = predicted_conf_scores.shape[0]
        tp = np.zeros(num_predictions)
        fp = np.zeros(num_predictions)

        for d in range(num_predictions):
            j_max = pred_gt_ho_assignment[d]
            if j_max >= 0:
                if not gt_assigned[j_max]:
                    tp[d] = 1
                    gt_assigned[j_max] = True
                else:
                    fp[d] = 1  # false positive (multiple detection)
            else:
                fp[d] = 1

        # compute precision/recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / num_hoi_gt_positives
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
