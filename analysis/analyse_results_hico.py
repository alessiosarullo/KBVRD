import argparse
import os
import pickle
import sys
from typing import List, Dict

from collections import Counter

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from analysis.utils import vis_one_image, plot_mat
from config import cfg
from lib.bbox_utils import compute_ious
from lib.dataset.hico.hico_split import HicoSplit, Splits
from lib.models.containers import Prediction
from lib.stats.evaluator import Evaluator
from lib.stats.utils import Timer, sort_and_filter, MetricFormatter

raise NotImplementedError
# TODO

try:
    matplotlib.use('Qt5Agg')
    # sys.argv[1:] = ['eval', '--save_dir', 'output/base/2019-07-10_13-35-46_vanilla']
    # sys.argv[1:] = ['eval', '--save_dir', 'output/base/2019-07-22_19-18-57_zsinters']

    # sys.argv[1:] = ['stats', '--save_dir', 'output/base/2019-07-10_10-17-30_vanilla']

    sys.argv[1:] = ['zs', '--save_dir', 'output/zss/2019-07-22_16-28-13_vanilla']

    # sys.argv[1:] = ['compare']
    # sys.argv[1:] = ['vis', '--save_dir', 'output/base/2019-06-05_17-43-04_vanilla']
except ImportError:
    pass


class Analyser:
    def __init__(self, dataset: HicoSplit, iou_thresh=0.5, hoi_score_thr=0.05):
        super().__init__()

        self.hoi_score_thr = hoi_score_thr

        self.dataset_split = dataset
        self.full_dataset = dataset.full_dataset

        self.act_conf_mat = np.zeros((self.dataset_split.num_actions, self.dataset_split.num_actions))

        self.ph_num_bins = 20
        self.ph_bins = np.arange(self.ph_num_bins + 1) / self.ph_num_bins
        self.prediction_act_hist = np.zeros((self.ph_bins.size, self.dataset_split.num_actions), dtype=np.int)

        self.num_gt = np.zeros((self.dataset_split.num_object_classes, self.dataset_split.num_actions))
        self.num_pred = np.zeros((self.dataset_split.num_object_classes, self.dataset_split.num_actions))

    def compute_stats(self, predictions: List[Dict]):
        assert len(predictions) == self.dataset_split.num_images, (len(predictions), self.dataset_split.num_images)

        gt_scores = self.full_dataset.split_annotations[self.dataset_split.split]
        predict_hoi_scores = np.full_like(gt_scores, fill_value=np.nan)
        for i, res in enumerate(predictions):
            prediction = Prediction(res)
            predict_hoi_scores[i, :] = prediction.hoi_scores
        gt_scores[gt_scores < 0] = 0

        hits = (predict_hoi_scores >= self.hoi_score_thr) & (gt_scores > 0)

        self.act_conf_mat /= np.maximum(1, self.act_conf_mat.sum(axis=1, keepdims=True))
        self.prediction_act_hist[-2, :] += self.prediction_act_hist[-1, :]
        self.prediction_act_hist = self.prediction_act_hist[:-1, :]

        self.pred_best_match_act_hist[-2, :] += self.pred_best_match_act_hist[-1, :]
        self.pred_best_match_act_hist = self.pred_best_match_act_hist[:-1, :]

def compute_cooccs(dataset: HicoDetSplit, gt_iou_thresh=0.5):
    act_cooccs = np.zeros((dataset.full_dataset.num_actions, dataset.full_dataset.num_actions))
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

    with open(cfg.prediction_file, 'rb') as f:
        results = pickle.load(f)
    cfg.load()
    return results


def _print_confidence_scores(analyser: Analyser, marked_preds=None):
    if marked_preds is not None:
        marked_preds = {x for x in marked_preds}
    else:
        marked_preds = []
    hicodet = analyser.dataset_split.full_dataset
    num_gt_act = np.sum(analyser.num_gt, axis=0).astype(np.int)
    print(' ' * 20,
          ' '.join([f'{x:6.2f}{"":7s}' for x in analyser.ph_bins[1:]]),
          '|', '{:>25s}'.format(f'>{analyser.ph_bins[1]}'),
          '|', f'{"#GT":>6s}')
    for j in range(hicodet.num_actions):
        pred_hist = analyser.prediction_act_hist[:, j]
        pred_match_hist = analyser.pred_best_match_act_hist[:, j]
        sum_p = np.sum(pred_hist[1:])
        sum_m = np.sum(pred_match_hist[1:])
        n_gt = num_gt_act[j]
        assert np.all(pred_match_hist[pred_hist == 0] == 0)
        print(f'{">" if j in marked_preds else "":1s}{hicodet.predicates[j]:19s}',
              ' '.join([f'{p:6d} ({"{:3.0f}{:1s}".format(100 * m / p, "%" if m > 0 else "") if p > 0 else "":>4s})'
                        for p, m in zip(pred_hist, pred_match_hist)]),
              '|', f'{sum_p:6d} (p: {"{:3.0f}%".format(100 * sum_m / sum_p) if sum_p > 0 else "":>4s}, r: {100 * sum_m / n_gt:3.0f}%)',
              '|', f'{n_gt:6d}')


def evaluate():
    results = _setup_and_load()
    hdtest = HicoDetSplitBuilder.get_split(HicoDetSplit, split=Splits.TEST)
    evaluator = Evaluator(dataset_split=hdtest, hoi_score_thr=None, num_hoi_thr=None)

    # evaluator.evaluate_predictions(results)
    evaluator.load(cfg.eval_res_file)

    hois = [inter for imdata in hdtest.full_dataset.split_data[Splits.TRAIN]
            for inter in hdtest.full_dataset.op_pair_to_interaction[imdata.box_classes[imdata.hois[:, 2]], imdata.hois[:, 1]]]
    hoi_hist = Counter(hois)
    non_rare_classes = np.array(sorted([c for c, n in hoi_hist.items() if n >= 10]))
    assert len(non_rare_classes) == 600 - 138

    map = evaluator.metrics['M-mAP']
    # print(np.mean(map))
    # nrand = 25
    # a, b = [], []
    # for t in range(10000):
    #     unseen = np.sort(np.random.choice(non_rare_classes, size=nrand, replace=False))
    #     a.append(unseen)
    #     b.append(np.mean(map[unseen]))
    # b = np.array(b)
    # inds = np.argsort(b)[::-1]
    # for i in inds[:10]:
    #     print(a[i])
    #     print(b[i])
    # [19  25 117 144 151 152 154 163 167 190 245 258 307 326 347 366 400 433 434 466 471 476 479 523 598]
    # 0.3231287508995482
    # [12  19  20  43  56 177 201 219 288 297 313 326 329 374 428 452 457 459 478 481 503 527 530 540 572]
    # 0.3185646326138225
    # [28  35  94 103 131 139 144 152 153 193 226 230 252 264 285 316 340 348 396 415 459 483 491 534 576]
    # 0.3152317659333818
    print(np.mean(map[np.array([19, 25, 117, 144, 151, 152, 154, 163, 167,
                                190, 245, 258, 307, 326, 347, 366, 400,
                                433, 434, 466, 471, 476, 479, 523, 598])]))

    # evaluator.output_metrics(sort=True, actions_to_keep=[1, 2, 84])
    # evaluator.output_metrics(sort=True, actions_to_keep=list(set(range(117)) - {84}))
    # evaluator.output_metrics(sort=True, actions_to_keep=list(range(117)))

    # stats = Evaluator_HD.evaluate_predictions(hdtest, results)
    # stats.print_metrics(sort=True)
    # Timer.print()


def compare():
    sys.argv[1:] = ['--save_dir', 'output/base/2019-07-02_16-06-32_vanilla']
    _setup_and_load()
    hds = HicoDetSplitBuilder.get_split(HicoDetSplit, split=Splits.TEST)
    evaluator = Evaluator(dataset_split=hds, hoi_score_thr=None, num_hoi_thr=None)
    evaluator.load(cfg.eval_res_file)
    obj_metrics1, hoi_metrics1, gt_obj_labels1, gt_hoi_labels1 = evaluator.output_metrics(sort=True, return_labels=True)

    print('=' * 100, '\n')

    sys.argv[1:] = ['--save_dir', 'output/bg/2019-07-04_17-59-58_margin-bgc10']
    _setup_and_load()
    hds = HicoDetSplitBuilder.get_split(HicoDetSplit, split=Splits.TEST)
    evaluator = Evaluator(dataset_split=hds, hoi_score_thr=None, num_hoi_thr=None)
    evaluator.load(cfg.eval_res_file)
    obj_metrics2, hoi_metrics2, gt_obj_labels2, gt_hoi_labels2 = evaluator.output_metrics(sort=True, return_labels=True)

    print('=' * 100, '\n')

    assert np.all(gt_hoi_labels2 == gt_hoi_labels1)
    c_hoi_metrics = {k: hoi_metrics2[k] - hoi_metrics1[k] for k in hoi_metrics2.keys()}
    gt_hoi_class_hist, _, hoi_class_inds = sort_and_filter(metrics=c_hoi_metrics,
                                                           gt_labels=gt_hoi_labels2,
                                                           all_classes=list(range(hds.full_dataset.num_interactions)),
                                                           sort=True)

    mf = MetricFormatter()
    mf.format_metric_and_gt_lines(gt_hoi_class_hist, c_hoi_metrics, hoi_class_inds, gt_str='GT HOIs')

    print('=' * 100, '\n')

    inds = np.argsort(c_hoi_metrics['M-mAP'])
    c_hoi_metrics = {k: v[inds] for k, v in c_hoi_metrics.items()}
    hoi_class_inds = [hoi_class_inds[i] for i in inds]
    gt_hoi_class_hist = {i: gt_hoi_class_hist[i] for i in inds}
    mf.format_metric_and_gt_lines(gt_hoi_class_hist, c_hoi_metrics, hoi_class_inds, gt_str='GT HOIs')

    # Timer.print()


def stats():
    results = _setup_and_load()
    res_save_path = cfg.res_stats_path
    os.makedirs(res_save_path, exist_ok=True)

    hdtrain = HicoDetSplitBuilder.get_split(HicoDetSplit, split=Splits.TRAIN)
    hicodet = hdtrain.full_dataset

    hdtest = HicoDetSplitBuilder.get_split(HicoDetSplit, split=Splits.TEST)

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

    with np.errstate(divide='ignore', invalid='ignore'):
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

    _print_confidence_scores(analyser)

    print('Recall:', np.mean(recall[num_gt > 0]))

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

    plot_mat(precision[obj_inds, :][:, pred_inds], s_predicates, s_objects, x_inds=pred_inds, y_inds=obj_inds, plot=False)
    plt.savefig(os.path.join(res_save_path, 'oa_prec.png'), dpi=300)

    plot_mat(recall[obj_inds, :][:, pred_inds], s_predicates, s_objects, x_inds=pred_inds, y_inds=obj_inds, plot=False)
    plt.savefig(os.path.join(res_save_path, 'oa_recall.png'), dpi=300)

    plot_mat(obj_hit_recall[obj_inds, :][:, pred_inds], s_predicates, s_objects, x_inds=pred_inds, y_inds=obj_inds, plot=False)
    plt.savefig(os.path.join(res_save_path, 'oa_obj_hit_recall.png'), dpi=300)

    plot_mat(spatial_match_recall[obj_inds, :][:, pred_inds], s_predicates, s_objects, x_inds=pred_inds, y_inds=obj_inds, plot=False)
    plt.savefig(os.path.join(res_save_path, 'oa_spatial_match_recall.png'), dpi=300)

    plot_mat((1 - recall)[obj_inds, :][:, pred_inds], s_predicates, s_objects, x_inds=pred_inds, y_inds=obj_inds, plot=False)
    plt.savefig(os.path.join(res_save_path, 'oa_misses.png'), dpi=300)

    # plt.show()


def zs_stats():
    results = _setup_and_load()
    res_save_path = cfg.res_stats_path
    os.makedirs(res_save_path, exist_ok=True)

    inds_dict = pickle.load(open(cfg.active_classes_file, 'rb'))
    seen_act_inds = sorted(inds_dict[Splits.TRAIN.value]['pred'].tolist())
    seen_obj_inds = sorted(inds_dict[Splits.TRAIN.value]['obj'].tolist())

    hdtrain = HicoDetSplitBuilder.get_split(HicoDetSplit, split=Splits.TRAIN, obj_inds=seen_obj_inds, pred_inds=seen_act_inds)
    hicodet = hdtrain.full_dataset
    seen_interactions = np.zeros((hicodet.num_object_classes, hicodet.num_actions), dtype=bool)
    seen_interactions[hdtrain.interactions[:, 1], hdtrain.interactions[:, 0]] = 1

    # hdval = HicoDetSplitBuilder.get_split(HicoDetSplit, split=Splits.VAL, obj_inds=seen_obj_inds, pred_inds=seen_act_inds)
    # print('Val only interactions:')
    # for p, o in hdval.interactions:
    #     if ~seen_interactions[o, p]:
    #         print(f'{hicodet.predicates[p]:20s} {hicodet.objects[o]:20s}')
    # print('Val only objects:', sorted(set(hdval.objects) - set(hdtrain.objects)))
    # print('Val only actions:', sorted(set(hdval.predicates) - set(hdtrain.predicates)))

    hdtest = HicoDetSplitBuilder.get_split(HicoDetSplit, split=Splits.TEST)
    unseen_interactions = np.zeros((hicodet.num_object_classes, hicodet.num_actions), dtype=bool)
    unseen_interactions[hdtest.interactions[:, 1], hdtest.interactions[:, 0]] = 1
    unseen_interactions[seen_interactions] = 0
    # print('Unseen interactions:')
    # for p, o in hdtest.interactions:
    #     if ~seen_interactions[o, p]:
    #         print(f'{hicodet.predicates[p]:20s} {hicodet.objects[o]:20s}')
    #         assert unseen_interactions[o, p]
    # print('Unseen objects:', sorted(set(hdtest.objects) - set(hdtrain.objects)))
    # print('Unseen actions:', sorted(set(hdtest.predicates) - set(hdtrain.predicates)))

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

    with np.errstate(divide='ignore', invalid='ignore'):
        num_gt, num_pred = analyser.num_gt, analyser.num_pred
        recall = analyser.gt_matches / num_gt
        precision = analyser.pred_matches / num_pred
        act_conf_mat = analyser.act_conf_mat

    obj_inds = np.argsort(num_gt.sum(axis=1))[::-1]
    s_objects = [hdtest.objects[i] for i in obj_inds]
    pred_inds = (np.argsort(num_gt.sum(axis=0)[1:])[::-1] + 1).tolist() + [0]  # no_interaction at the end
    s_predicates = [hdtest.predicates[i] for i in pred_inds]

    _print_confidence_scores(analyser, marked_preds=set(range(hicodet.num_actions)) - set(seen_act_inds))

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
    print('#' * 30, 'Zero shot confusion matrix (ZS to all)', '#' * 30, '\n')
    print(zero_shot_cmat_str)

    zs_confmat_inv = act_conf_mat[pred_inds_zs, :][:, unseen_pred_inds]
    zs_confmat_inv[zs_confmat_inv == 0] = np.inf
    plot_mat(zs_confmat_inv, zs_predicates, s_predicates_zs, x_inds=unseen_pred_inds, y_inds=pred_inds_zs, plot=False)
    plt.savefig(os.path.join(res_save_path, 'zs_conf_mat_inv.png'), dpi=300)
    zero_shot_cmat_str = '\n'.join(['%-20s %-20s %.3f' % (s_predicates_zs[p], zs_predicates[zsp], zs_confmat_inv[p, zsp])
                                    for p, zsp in np.stack(np.where(~np.isinf(zs_confmat_inv) & (zs_confmat_inv > 0)), axis=1)])
    print()
    print('#' * 100, '\n')
    print('#' * 30, 'Zero shot confusion matrix (all to ZS)', '#' * 30, '\n')
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

    plot_mat(precision[obj_inds, :][:, unseen_pred_inds].T, s_objects, zs_predicates, x_inds=obj_inds, y_inds=unseen_pred_inds, plot=False)
    plt.savefig(os.path.join(res_save_path, 'zs_ao_prec.png'), dpi=300)
    plot_mat(recall[obj_inds, :][:, unseen_pred_inds].T, s_objects, zs_predicates, x_inds=obj_inds, y_inds=unseen_pred_inds, plot=False)
    plt.savefig(os.path.join(res_save_path, 'zs_ao_rec.png'), dpi=300)

    # plt.show()


def visualise_images():
    act_thr = 0.1

    results = _setup_and_load()
    hds = HicoDetSplitBuilder.get_split(HicoDetSplit, split=Splits.TEST)  # type: HicoDetSplit

    output_dir = os.path.join('analysis', 'output', 'vis', *(cfg.output_path.split('/')[1:]))
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
             'zs': zs_stats,
             'eval': evaluate,
             'compare': compare,
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