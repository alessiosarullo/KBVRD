import argparse
import os
import pickle
import sys
from typing import List, Dict

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from analysis.utils import vis_one_image, plot_mat
from config import cfg
from lib.bbox_utils import compute_ious
from lib.dataset.hicodet.hicodet_split import HicoDetSplitBuilder, HicoDetSplit, Splits, Example, Minibatch
from lib.models.containers import Prediction
from lib.stats.evaluator import Evaluator
from lib.stats.utils import Timer

try:
    matplotlib.use('Qt5Agg')
    # sys.argv[1:] = ['eval', '--save_dir', 'output/zsgc/2019-06-24_15-37-24_att']
    # sys.argv[1:] = ['stats', '--save_dir', 'output/zsgc/2019-07-01_22-15-33_bare']
    # sys.argv[1:] = ['stats', '--save_dir', 'output/zsgc/2019-07-02_20-50-20_att']
    # sys.argv[1:] = ['stats', '--save_dir', 'output/zsgc/2019-07-02_17-12-17_bare_softl']
    sys.argv[1:] = ['stats', '--save_dir', 'output/zsgc/2019-07-04_10-29-22_bare_softl01']
    # sys.argv[1:] = ['vis', '--save_dir', 'output/base/2019-06-05_17-43-04_vanilla']
except ImportError:
    pass


class Analyser:
    def __init__(self, dataset: HicoDetSplit, iou_thresh=0.5, hoi_score_thr=0.5):
        super().__init__()

        self.iou_thresh = iou_thresh
        self.hoi_score_thr = hoi_score_thr

        self.dataset = dataset
        # self.gt_interactions = []  # (p, o)
        # self.predict_ho_obj_scores = []
        # self.predict_action_scores = []
        # self.predict_gt_assignment_unconstrained = []

        # NOTE: this sums up to the number of predictions.
        # Actually, it does not. I am not considering predictions that do not match any GT entry because of the sparsity of annotations. If
        # required, those should be considered mistakes of the null interaction.
        self.act_conf_mat = np.zeros((self.dataset.num_predicates, self.dataset.num_predicates))

        self.gt_matches = np.zeros((self.dataset.num_object_classes, self.dataset.num_predicates))
        self.pred_matches = np.zeros((self.dataset.num_object_classes, self.dataset.num_predicates))
        self.gt_spatial_matches = np.zeros((self.dataset.num_object_classes, self.dataset.num_predicates))
        self.gt_candidate_matches = np.zeros((self.dataset.num_object_classes, self.dataset.num_predicates))
        self.num_gt = np.zeros((self.dataset.num_object_classes, self.dataset.num_predicates))
        self.num_pred = np.zeros((self.dataset.num_object_classes, self.dataset.num_predicates))

    def compute_stats(self, predictions: List[Dict]):
        assert len(predictions) == self.dataset.num_images, (len(predictions), self.dataset.num_images)

        for i, res in enumerate(predictions):
            ex = self.dataset.get_img_entry(i, read_img=False)
            prediction = Prediction(res)
            self.process_prediction(i, ex, prediction)

        self.act_conf_mat /= np.maximum(1, self.act_conf_mat.sum(axis=1, keepdims=True))

    def process_prediction(self, im_id, gt_entry: Example, prediction: Prediction):
        if isinstance(gt_entry, Example):
            gt_hoi_triplets = gt_entry.gt_hois[:, [0, 2, 1]]  # (h, o, i)
            gt_interactions = np.stack([gt_hoi_triplets[:, 2], gt_entry.gt_obj_classes[gt_hoi_triplets[:, 1]]], axis=1)  # (a, o)
            num_gt_hois = gt_hoi_triplets.shape[0]

            gt_boxes = gt_entry.gt_boxes.astype(np.float, copy=False)
        else:
            raise ValueError('Unknown type for GT entry: %s.' % str(type(gt_entry)))

        predict_action_scores = np.zeros([0, self.dataset.num_predicates])
        predict_ho_obj_scores = np.zeros([0, self.dataset.num_object_classes])
        predict_ho_pairs = np.zeros((0, 2), dtype=np.int)
        predict_boxes = np.zeros((0, 4))
        if prediction.obj_boxes is not None:
            assert prediction.obj_im_inds.shape[0] == prediction.obj_boxes.shape[0]

            predict_boxes = prediction.obj_boxes

            if prediction.ho_pairs is not None:
                assert len(np.unique(prediction.obj_im_inds)) == len(np.unique(prediction.ho_img_inds)) == 1

                predict_ho_pairs = prediction.ho_pairs
                assert prediction.hoi_scores is None
                assert prediction.obj_im_inds.shape[0] == prediction.obj_scores.shape[0]
                assert prediction.action_scores is not None

                predict_action_scores = prediction.action_scores
                predict_ho_obj_scores = prediction.obj_scores[predict_ho_pairs[:, 1], :]
        else:
            assert prediction.ho_pairs is None

        num_predictions = predict_ho_pairs.shape[0]

        pred_gt_ious = compute_ious(predict_boxes, gt_boxes)
        pred_ho_objects = np.argmax(predict_ho_obj_scores, axis=1)

        #################################################################################
        # Confusion matrix & co                                                         #
        #################################################################################

        # First, find predictions that 1) match spatially with GT and 2) classify the object correctly.
        pred_gt_spatial_matches = np.zeros((num_predictions, num_gt_hois), dtype=bool)
        pred_gt_candidate_matches = np.zeros((num_predictions, num_gt_hois))
        for predict_idx, (ph_ind, po_ind) in enumerate(predict_ho_pairs):
            gt_pair_ious = np.zeros(num_gt_hois)
            for gtidx, (gh_ind, go_ind, ga) in enumerate(gt_hoi_triplets):
                iou_h = pred_gt_ious[ph_ind, gh_ind]
                iou_o = pred_gt_ious[po_ind, go_ind]
                gt_pair_ious[gtidx] = min(iou_h, iou_o)

            spatial_matches = (gt_pair_ious >= self.iou_thresh)
            po = pred_ho_objects[predict_idx]
            matches = np.flatnonzero((gt_interactions[:, 1] == po) & spatial_matches)

            pred_gt_spatial_matches[predict_idx, spatial_matches] = 1
            pred_gt_candidate_matches[predict_idx, matches] = gt_pair_ious[matches]

            self.num_pred[po, predict_action_scores[predict_idx, :] >= self.hoi_score_thr] += 1

        # Then decide which action predictions are hits. For each pair, an action is considered predicted if the assigned score satisfies some
        # criterion (e.g., is greater than a certain threshold). An action prediction is a hit if there is a matching GT (in the sense defined above)
        # with the same action (note: duplicates are NOT allowed, i.e., only the best match for a GT elements is considered correct).
        pred_has_been_matched = np.zeros(num_predictions, dtype=bool)
        hits = np.zeros_like(predict_action_scores)
        best_match_for_gt = np.full(num_gt_hois, fill_value=-1, dtype=np.int)
        for gtidx, (ga, go) in enumerate(gt_interactions):
            self.num_gt[go, ga] += 1

            if np.any(pred_gt_spatial_matches[:, gtidx]):
                self.gt_spatial_matches[go, ga] += 1

            pred_candidate_ious = pred_gt_candidate_matches[:, gtidx]
            pred_candidates = (pred_candidate_ious > 0)

            if np.any(pred_candidates):
                assert np.all(pred_candidate_ious[pred_candidates] >= self.iou_thresh)
                self.gt_candidate_matches[go, ga] += 1

                best_match = np.argmax(pred_candidate_ious)
                best_match_for_gt[gtidx] = best_match
                if predict_action_scores[best_match, ga] >= self.hoi_score_thr:
                    hits[best_match, ga] += 1
                    self.gt_matches[go, ga] += 1
                    self.act_conf_mat[ga, ga] += 1
                    if ~pred_has_been_matched[best_match]:
                        self.pred_matches[go, ga] += 1
        hits = (hits > 0)

        # Finally, fill the off-diagonal elements of the confusion matrix. To do this, find the best match for any given GT triplet and consider
        # false positives all its predicted actions that are not hits (see above). Another for loop is required because the GT is not multi-label.
        for gtidx, (ga, go) in enumerate(gt_interactions):
            best_match = best_match_for_gt[gtidx]
            if best_match >= 0:
                predicted_acts = (predict_action_scores[best_match, :] >= self.hoi_score_thr)
                if ~hits[best_match, ga]:
                    prediction_misses = predicted_acts & ~hits[best_match, :]
                    assert not prediction_misses[ga]
                    self.act_conf_mat[ga, prediction_misses] += 1

        # # Uncomment this if it is desired to considered unmatched detections as misses of the null interactions.
        # matches_set = set(np.flatnonzero(best_match_for_gt >= 0).tolist())
        # for predict_idx in range(num_predictions):
        #     if predict_idx not in matches_set:
        #         assert not np.any(hits[predict_idx, :])
        #         predicted_acts = (predict_action_scores[predict_idx, :] >= self.hoi_score_thr)
        #         self.act_conf_mat[0, predicted_acts] += 1


def compute_cooccs(dataset: HicoDetSplit, gt_iou_thresh=0.5):
    act_cooccs = np.zeros((dataset.hicodet.num_predicates, dataset.hicodet.num_predicates))
    for gt_idx in range(len(dataset)):
        gt_entry = dataset.get_img_entry(gt_idx, read_img=False)
        gt_hoi_triplets = gt_entry.gt_hois[:, [0, 2, 1]]  # (h, o, i)
        gt_interactions = np.stack([gt_hoi_triplets[:, 2], gt_entry.gt_obj_classes[gt_hoi_triplets[:, 1]]], axis=1)  # (a, o)
        num_gt_hois = gt_hoi_triplets.shape[0]

        gt_boxes = gt_entry.gt_boxes.astype(np.float, copy=False)

        #################################################################################
        # Action co-occurrences in GT                                                   #
        #################################################################################

        gt_gt_ious = compute_ious(gt_boxes, gt_boxes)
        gt_gt_pair_ious = np.zeros((num_gt_hois, num_gt_hois))
        for i1, (h1_ind, o1_ind, a1) in enumerate(gt_hoi_triplets):
            for i2, (h2_ind, o2_ind, a2) in enumerate(gt_hoi_triplets):
                iou_h = gt_gt_ious[h1_ind, h2_ind]
                iou_o = gt_gt_ious[o1_ind, o2_ind]
                gt_gt_pair_ious[i1, i2] = min(iou_h, iou_o)

        assert np.all(gt_gt_pair_ious == gt_gt_pair_ious.T)
        for i1, (a1, o1) in enumerate(gt_interactions):
            overlaps = (gt_gt_pair_ious[i1, :] >= gt_iou_thresh)
            same_object = (o1 == gt_interactions[:, 1])
            match = overlaps & same_object
            assert match[i1], (i1, match)
            matching_idxs = np.flatnonzero(match)
            matching_idxs = matching_idxs[matching_idxs > i1]  # only examine upper triangle
            for i2 in matching_idxs:
                a2 = gt_interactions[i2, 0]
                act_cooccs[a1, a2] += 1
    return act_cooccs


def _setup_and_load():
    cfg.parse_args(fail_if_missing=False, reset=True)

    with open(cfg.program.prediction_file, 'rb') as f:
        results = pickle.load(f)
    cfg.load()
    return results


def evaluate():
    results = _setup_and_load()
    hds = HicoDetSplitBuilder.get_split(HicoDetSplit, split=Splits.TEST)
    evaluator = Evaluator(dataset_split=hds, hoi_score_thr=None, num_hoi_thr=None)

    # evaluator.evaluate_predictions(results)
    evaluator.load(cfg.program.eval_res_file)

    evaluator.output_metrics(sort=True, actions_to_keep=[1, 2, 84])
    # evaluator.output_metrics(sort=True, actions_to_keep=list(set(range(117)) - {84}))
    # evaluator.output_metrics(sort=True, actions_to_keep=list(range(117)))

    # stats = Evaluator_HD.evaluate_predictions(hds, results)
    # stats.print_metrics(sort=True)
    # Timer.print()


def hist():
    predictions = _setup_and_load()
    hds = HicoDetSplitBuilder.get_split(HicoDetSplit, split=Splits.TEST)

    hist = np.zeros(11)

    for i, res in enumerate(predictions):
        prediction = Prediction(res)
        if prediction.ho_pairs is not None:
            h, bins = np.histogram(prediction.action_scores, bins=11)
            hist += h

    for j


def stats():
    results = _setup_and_load()
    res_save_path = cfg.program.res_stats_path
    os.makedirs(res_save_path, exist_ok=True)

    seen_act_inds = sorted(pickle.load(open(cfg.program.active_classes_file, 'rb'))[Splits.TRAIN.value]['pred'].tolist())
    seen_obj_inds = cfg.data.obj_inds
    hdtrain = HicoDetSplitBuilder.get_split(HicoDetSplit, split=Splits.TRAIN, obj_inds=seen_obj_inds, pred_inds=seen_act_inds)
    hicodet = hdtrain.hicodet
    seen_interactions = np.zeros((hicodet.num_object_classes, hicodet.num_predicates), dtype=bool)
    seen_interactions[hdtrain.interactions[:, 1], hdtrain.interactions[:, 0]] = 1

    hdval = HicoDetSplitBuilder.get_split(HicoDetSplit, split=Splits.VAL, obj_inds=seen_obj_inds, pred_inds=seen_act_inds)
    print('Val only interactions:')
    for p, o in hdval.interactions:
        if ~seen_interactions[o, p]:
            print(f'{hicodet.predicates[p]:20s} {hicodet.objects[o]:20s}')
    print('Val only objects:', sorted(set(hdval.objects) - set(hdtrain.objects)))
    print('Val only actions:', sorted(set(hdval.predicates) - set(hdtrain.predicates)))

    hdtest = HicoDetSplitBuilder.get_split(HicoDetSplit, split=Splits.TEST)
    unseen_interactions = np.zeros((hicodet.num_object_classes, hicodet.num_predicates), dtype=bool)
    unseen_interactions[hdtest.interactions[:, 1], hdtest.interactions[:, 0]] = 1
    unseen_interactions[seen_interactions] = 0
    print('Unseen interactions:')
    for p, o in hdtest.interactions:
        if ~seen_interactions[o, p]:
            print(f'{hicodet.predicates[p]:20s} {hicodet.objects[o]:20s}')
            assert unseen_interactions[o, p]
    print('Unseen objects:', sorted(set(hdtest.objects) - set(hdtrain.objects)))
    print('Unseen actions:', sorted(set(hdtest.predicates) - set(hdtrain.predicates)))

    cache_fn = os.path.join(res_save_path, 'cache.pkl')
    try:
        with open(cache_fn, 'rb') as f:
            analyser = pickle.load(f)
        print('Loaded')
    except FileNotFoundError:
        analyser = Analyser(dataset=hdtest)
        analyser.compute_stats(results)
        with open(cache_fn, 'wb') as f:
            pickle.dump(analyser, f)
        print('Saved')

    num_gt, num_pred = analyser.num_gt, analyser.num_pred
    recall = analyser.gt_matches / num_gt
    precision = analyser.pred_matches / num_pred
    obj_hit_recall = analyser.gt_candidate_matches / num_gt
    spatial_match_recall = analyser.gt_spatial_matches / num_gt
    act_conf_mat = analyser.act_conf_mat

    obj_inds = np.argsort(num_gt.sum(axis=1))[::-1]
    s_objects = [hdtest.objects[i] for i in obj_inds]
    pred_inds = (np.argsort(num_gt.sum(axis=0)[1:])[::-1] + 1).tolist() + [0]  # no_interaction at the end
    s_predicates = [hdtest.predicates[i] for i in pred_inds]

    print(np.mean(recall[num_gt > 0]))

    zero_shot_preds = np.full_like(num_pred, fill_value=np.inf)
    zero_shot_preds[seen_interactions] = -1
    zero_shot_preds[unseen_interactions] = num_pred[unseen_interactions]
    plot_mat(zero_shot_preds, hdtest.predicates, hdtest.objects, plot=False, neg_color=[0.5, 0.5, 0.5, 1], zero_color=[0.8, 0, 0.8, 1], log=True)
    plt.savefig(os.path.join(res_save_path, 'zero_shot.png'), dpi=300)
    zero_shot_str = '\n'.join(['%-20s %-20s %d' % (hdtest.predicates[p], hdtest.objects[o], zero_shot_preds[o, p])
                               for p, o in np.stack(np.where(~np.isinf(zero_shot_preds.T) & (zero_shot_preds.T >= 0)), axis=1)])
    print()
    print('#' * 100, '\n')
    print('#' * 30, 'Zero shot', '#' * 30, '\n')
    print(zero_shot_str)
    with open(os.path.join(res_save_path, 'zero_shot.txt'), 'w') as f:
        f.write(zero_shot_str)

    unseen_pred_inds = np.array(sorted(set(hdtest.active_predicates.tolist()) - set(hdtrain.active_predicates.tolist())))
    pred_inds_zs = unseen_pred_inds.tolist() + [p for p in pred_inds if p not in unseen_pred_inds]
    s_predicates_zs = [hdtest.predicates[i] for i in pred_inds_zs]
    zs_predicates = [hdtest.predicates[i] for i in unseen_pred_inds]
    zs_confmat = act_conf_mat[unseen_pred_inds, :][:, pred_inds_zs]
    zs_confmat[zs_confmat == 0] = np.inf
    plot_mat(zs_confmat, s_predicates_zs, zs_predicates, x_inds=pred_inds_zs, y_inds=unseen_pred_inds, plot=False)
    plt.savefig(os.path.join(res_save_path, 'zs_conf_mat.png'), dpi=300)
    zero_shot_cmat_str = '\n'.join(['%-20s %-20s %.3f' % (zs_predicates[zsp], s_predicates_zs[p], zs_confmat[zsp, p])
                                    for zsp, p in np.stack(np.where(~np.isinf(zs_confmat) & (zs_confmat > 0)), axis=1)])
    print()
    print('#' * 100, '\n')
    print('#' * 30, 'Zero shot confusion matrix', '#' * 30, '\n')
    print(zero_shot_cmat_str)

    # act_cooccs_train = compute_cooccs(hdtrain)
    # zs_act_cooccs_train = act_cooccs_train[unseen_pred_inds, :][:, pred_inds_zs]
    # zs_act_cooccs_train[zs_act_cooccs_train == 0] = np.inf
    # plot_mat(zs_act_cooccs_train, s_predicates_zs, zs_predicates, x_inds=pred_inds_zs, y_inds=unseen_pred_inds, plot=False, log=True)
    # plt.savefig(os.path.join(res_save_path, 'zs_coocc_mat_train.png'), dpi=300)

    act_cooccs_test = compute_cooccs(hdtest)
    zs_act_cooccs_test = act_cooccs_test[unseen_pred_inds, :][:, pred_inds_zs]
    zs_act_cooccs_test[zs_act_cooccs_test == 0] = np.inf
    plot_mat(zs_act_cooccs_test, s_predicates_zs, zs_predicates, x_inds=pred_inds_zs, y_inds=unseen_pred_inds, plot=False, log=True)
    plt.savefig(os.path.join(res_save_path, 'zs_coocc_mat_test.png'), dpi=300)
    zero_shot_coocc_str = '\n'.join(['%-20s %-20s %.3f' % (zs_predicates[zsp], s_predicates_zs[p], zs_act_cooccs_test[zsp, p])
                                     for zsp, p in np.stack(np.where(~np.isinf(zs_act_cooccs_test) & (zs_act_cooccs_test > 0)), axis=1)])
    print()
    print('#' * 100, '\n')
    print('#' * 30, 'Zero shot co-occurrences matrix', '#' * 30, '\n')
    print(zero_shot_coocc_str)

    plot_mat(precision[obj_inds, :][:, pred_inds], s_predicates, s_objects, x_inds=pred_inds, y_inds=obj_inds, plot=False)
    plt.savefig(os.path.join(res_save_path, 'oa_prec.png'), dpi=300)
    plot_mat(precision[obj_inds, :][:, unseen_pred_inds].T, s_objects, zs_predicates, x_inds=obj_inds, y_inds=unseen_pred_inds, plot=False)
    plt.savefig(os.path.join(res_save_path, 'zs_ao_prec.png'), dpi=300)
    plot_mat(recall[obj_inds, :][:, unseen_pred_inds].T, s_objects, zs_predicates, x_inds=obj_inds, y_inds=unseen_pred_inds, plot=False)
    plt.savefig(os.path.join(res_save_path, 'zs_ao_rec.png'), dpi=300)

    # exit(0)  # FIXME

    out_of_gt_preds = (num_gt == 0).astype(np.float) * num_pred
    out_of_gt_preds[out_of_gt_preds == 0] = np.inf
    plot_mat(out_of_gt_preds, hdtest.predicates, hdtest.objects, plot=False)
    plt.savefig(os.path.join(res_save_path, 'out_of_gt.png'), dpi=300)
    out_of_gt_str = '\n'.join(['%-20s %-20s %d' % (hdtest.predicates[p], hdtest.objects[o], out_of_gt_preds[o, p])
                               for p, o in np.stack(np.where(~np.isinf(out_of_gt_preds.T)), axis=1)])
    print()
    print('#' * 100, '\n')
    # print('#' * 30, 'Out of GT', '#' * 30, '\n')
    # print(out_of_gt_str)
    with open(os.path.join(res_save_path, 'out_of_gt.txt'), 'w') as f:
        f.write(out_of_gt_str)

    confmat = act_conf_mat[pred_inds, :][:, pred_inds]
    confmat[confmat == 0] = np.inf
    plot_mat(confmat, s_predicates, s_predicates, x_inds=pred_inds, y_inds=pred_inds, plot=False)
    plt.savefig(os.path.join(res_save_path, 'conf_mat.png'), dpi=300)

    plot_mat(recall[obj_inds, :][:, pred_inds], s_predicates, s_objects, x_inds=pred_inds, y_inds=obj_inds, plot=False)
    plt.savefig(os.path.join(res_save_path, 'oa_recall.png'), dpi=300)

    plot_mat(obj_hit_recall[obj_inds, :][:, pred_inds], s_predicates, s_objects, x_inds=pred_inds, y_inds=obj_inds, plot=False)
    plt.savefig(os.path.join(res_save_path, 'oa_obj_hit_recall.png'), dpi=300)

    plot_mat(spatial_match_recall[obj_inds, :][:, pred_inds], s_predicates, s_objects, x_inds=pred_inds, y_inds=obj_inds, plot=False)
    plt.savefig(os.path.join(res_save_path, 'oa_spatial_match_recall.png'), dpi=300)

    plot_mat((1 - recall)[obj_inds, :][:, pred_inds], s_predicates, s_objects, x_inds=pred_inds, y_inds=obj_inds, plot=False)
    plt.savefig(os.path.join(res_save_path, 'oa_misses.png'), dpi=300)

    # plt.show()


def visualise_images():
    act_thr = 0.1

    results = _setup_and_load()
    hds = HicoDetSplitBuilder.get_split(HicoDetSplit, split=Splits.TEST)

    output_dir = os.path.join('analysis', 'output', 'vis', *(cfg.program.output_path.split('/')[1:]))
    os.makedirs(output_dir, exist_ok=True)

    for b_idx in range(len(hds)):
        entry = hds.get_img_entry(b_idx, read_img=False)  # type: Example
        im_fn = entry.filename

        prediction_dict = results[b_idx]
        prediction = Prediction(prediction_dict)

        boxes = prediction.obj_boxes
        obj_scores = prediction.obj_scores
        ho_pairs = prediction.ho_pairs
        act_scores = prediction.action_scores
        if obj_scores is None:
            continue

        box_classes = np.argmax(obj_scores, axis=1)
        box_class_scores = obj_scores[np.arange(boxes.shape[0]), box_classes]

        # if im_fn not in [s.strip() for s in """
        # HICO_test2015_00000003.jpg
        # """.split('\n')]:
        #     continue

        # FIND
        # if act_scores[box_classes[ho_pairs[:, 1]] == 18, 9].size == 0 or np.all(act_scores[box_classes[ho_pairs[:, 1]] == 18, 9] < act_thr):
        #     continue
        # print(im_fn)

        im = cv2.imread(os.path.join(hds.img_dir, im_fn))
        assert np.all(boxes[:, [0, 2]] < im.shape[1]) and np.all(boxes[:, [1, 3]] < im.shape[0]), b_idx
        continue
        vis_one_image(
            hds, im[:, :, [2, 1, 0]],  # BGR -> RGB for visualization
            boxes=boxes, box_classes=box_classes, box_classes_scores=box_class_scores, masks=None,
            ho_pairs=ho_pairs, action_class_scores=act_scores,
            output_file_path=os.path.join(output_dir, os.path.splitext(im_fn)[0]),
            ext='png',
            act_thr=act_thr,
            dpi=400, fontsize=2,
        )

        if b_idx >= 0:
            break


def main():
    funcs = {'vis': visualise_images,
             'stats': stats,
             'eval': evaluate,
             'hist': hist,
             }
    print(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('func', type=str, choices=funcs.keys())
    namespace = parser.parse_known_args()
    func = vars(namespace[0])['func']
    sys.argv = sys.argv[:1] + namespace[1]
    print(sys.argv)
    funcs[func]()


if __name__ == '__main__':
    main()
