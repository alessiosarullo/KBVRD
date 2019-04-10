import argparse
import os
import pickle
import sys
from collections import Counter
from typing import Dict

import cv2
import matplotlib
import numpy as np
import torch
from matplotlib import gridspec
from matplotlib import pyplot as plt

from analysis.utils import vis_one_image, plot_mat, heatmap, annotate_heatmap
from config import cfg
from lib.dataset.hicodet import HicoDetInstanceSplit, Splits
from lib.dataset.utils import Minibatch, get_counts
from lib.knowledge_extractors.imsitu_knowledge_extractor import ImSituKnowledgeExtractor
from lib.models.utils import Prediction
from lib.stats.evaluator import Evaluator
from scripts.utils import get_all_models_by_name

matplotlib.use('Qt5Agg')


def _setup_and_load():
    cfg.parse_args(allow_required=False, reset=True)

    with open(cfg.program.result_file, 'rb') as f:
        results = pickle.load(f)
    cfg.load()
    # cfg.program.load_precomputed_feats = False
    return results


def vrd_style_eval_count():
    results = _setup_and_load()
    hds = HicoDetInstanceSplit.get_split(split=Splits.TEST)
    stats = Evaluator.evaluate_predictions(hds, results)
    stats.print_metrics()

    gt_hois = hds.hois
    gt_hoi_hist = Counter(gt_hois[:, 1])
    num_gt_hois = sum(gt_hoi_hist.values())
    assert num_gt_hois == gt_hois.shape[0]
    num_pred_hois = stats.num_hoi_predictions_per_class
    print('  GT HOIs: [%s]' % ', '.join(['%s (%3.0f%%)' % (c.replace('_', ' ').strip(), 100 * gt_hoi_hist[i] / num_gt_hois)
                                         for i, c in enumerate(hds.predicates)]))
    print('Pred HOIs: [%s]' % ', '.join(['%s (%3.0f%%)' % (c.replace('_', ' ').strip(), 100 * num_pred_hois[i] / np.sum(num_pred_hois))
                                         for i, c in enumerate(hds.predicates)]))

    obj_class_hist = Counter(hds.obj_labels)
    num_gt_obj_classes = sum(obj_class_hist.values())
    assert num_gt_obj_classes == hds.obj_labels.shape[0]
    num_pred_objs = stats.num_obj_predictions_per_class
    print('  GT objects: [%s]' % ', '.join(['%s (%3.0f%%)' % (c.replace('_', ' ').strip(), 100 * obj_class_hist[i] / num_gt_obj_classes)
                                            for i, c in enumerate(hds.objects)]))
    print('Pred objects: [%s]' % ', '.join(['%s (%3.0f%%)' % (c.replace('_', ' ').strip(), 100 * num_pred_objs[i] / np.sum(num_pred_objs))
                                            for i, c in enumerate(hds.objects)]))


def stats():
    base_argv = sys.argv
    exps = [
        # 'output/hoi/2019-04-03_11-21-41_vanilla',
        'output/zero/2019-04-03_10-28-10_vanilla',
        # 'output/kb/2019-04-09_11-47-18_ds-imsitu'
    ]
    true_pos = []
    pos = None
    for exp in exps:
        sys.argv = base_argv + ['--save_dir', exp]
        print('=' * 100, '\n', sys.argv)

        results = _setup_and_load()

        hdtrain = HicoDetInstanceSplit.get_split(split=Splits.TRAIN)
        hdtest = HicoDetInstanceSplit.get_split(split=Splits.TEST)

        stats = Evaluator.evaluate_predictions(hdtest, results)
        stats.print_metrics(sort=True)

        # detector = get_all_models_by_name()[cfg.program.model](hdtrain)
        # ckpt = torch.load(cfg.program.saved_model_file, map_location='cpu')
        # detector.load_state_dict(ckpt['state_dict'])

        # op_adj_mat = detector.hoi_branch.op_adj_mat.squeeze(dim=-1).detach().numpy()
        # op_conf_mat = torch.sigmoid(detector.hoi_branch.op_conf_mat.squeeze(dim=-1).detach()).numpy()
        #
        # ds_counts = get_counts(dataset=hdtrain)
        # ds_counts[:, 0] = 0  # exclude null interaction
        # ds = np.minimum(1, ds_counts)
        # imsitu_counts = ImSituKnowledgeExtractor().extract_freq_matrix(hdtrain)
        # imsitu_counts[:, 0] = 0  # exclude null interaction
        # imsitu = np.minimum(1, imsitu_counts)  # only check if the pair exists (>=1 occurrence) or not (0 occurrences)

        # # Plot
        # plt.figure(figsize=(16, 9))

        # gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1],
        #                        wspace=0.01, hspace=0.4, top=0.9, bottom=0.1, left=0.05, right=0.95)
        # plot_mat(op_adj_mat[:, :, 1] / 3 + ds * 2 / 3, predicates, objects, axes=plt.subplot(gs[0, 0]))
        # plot_mat(op_conf_mat[:, :, 1], predicates, objects, axes=plt.subplot(gs[0, 1]))
        # plot_mat(op_adj_mat[:, :, 0] / 3 + imsitu * 2 / 3, predicates, objects, axes=plt.subplot(gs[1, 0]))
        # plot_mat(op_conf_mat[:, :, 0], predicates, objects, axes=plt.subplot(gs[1, 1]))
        # plt.show()


        pred_thr = 0.5
        gt_hois = np.concatenate(stats.hoi_labels, axis=0)
        gt_hoi_objs = np.concatenate(stats.hoi_obj_labels, axis=0)
        predictions = np.concatenate(stats.hoi_predictions, axis=0)
        assert gt_hois.shape[0] == gt_hoi_objs.shape[0] == predictions.shape[0] and gt_hois.shape[1] == predictions.shape[1] == hdtrain.num_predicates
        tps = np.zeros([hdtrain.num_object_classes, hdtrain.num_predicates])
        gt_counts = np.zeros_like(tps)
        for gth, gto, ph in zip(gt_hois, gt_hoi_objs, predictions):
            gth_inds = np.flatnonzero(gth)
            tps[gto, gth_inds] += (ph[gth_inds] > pred_thr)
            gt_counts[gto, gth_inds] += 1
        assert np.all(tps <= gt_counts)
        if pos is None:
            pos = gt_counts
        assert np.all(pos == gt_counts)
        true_pos.append(tps)

    obj_inds = np.argsort(pos.sum(axis=1))[::-1]
    pred_inds = np.argsort(pos.sum(axis=0))[::-1]
    true_pos = np.stack(true_pos, axis=2)

    assert true_pos.shape[2] == 2
    mat = (true_pos[:, :, 0] - true_pos[:, :, 1]) / pos
    mat = mat / 2 + 0.5
    plot_mat(mat[:, pred_inds][obj_inds, :], [hdtrain.predicates[i] for i in pred_inds], [hdtrain.objects[i] for i in obj_inds],
             x_inds=pred_inds, y_inds=obj_inds,
             cbar=True, bin_colours=True, plot=False)
    plt.show()


def att():
    results = _setup_and_load()
    hds = HicoDetInstanceSplit.get_split(split=Splits.TEST)
    stats = Evaluator.evaluate_predictions(hds, results)
    stats.print_metrics()

    with open(cfg.program.watched_values_file, 'rb') as f:
        watched_values = pickle.load(f)  # type: Dict
    assert len(watched_values) == 1
    im_att_orig = list(watched_values.values())[0]

    assert len(im_att_orig) == len(stats.hoi_labels) == len(stats.hoi_obj_labels) == len(stats.hoi_gt_pred_assignment)
    im_att = []
    for iv, igt2p in zip(im_att_orig, stats.hoi_gt_pred_assignment):
        if igt2p.size > 0:
            x = np.array([iv[gt2p, :] if gt2p >= 0 else [0] for gt2p in igt2p])
            im_att.append(x)
    att = np.concatenate(im_att, axis=0)
    hoi_labels = np.concatenate(stats.hoi_labels, axis=0)
    hoi_obj_labels = np.concatenate(stats.hoi_obj_labels, axis=0).astype(np.int)
    assert att.shape[0] == hoi_labels.shape[0] == hoi_obj_labels.shape[0]

    mat = np.zeros((hds.num_object_classes, hds.num_predicates, 1))
    mat2 = np.zeros((hds.num_object_classes, hds.num_predicates))
    for i in range(att.shape[0]):
        pred_att = att[i, :]
        obj = hoi_obj_labels[i]
        pred = np.flatnonzero(hoi_labels[i, :])
        for p in pred:
            mat[obj, p, :] += pred_att
            mat2[obj, p] += 1
    # assert np.sum(mat2 > 0) == 600, np.sum(mat2 > 0)  # FIXME it's 599, one is missing

    mat = mat / np.maximum(1, mat2[:, :, None])

    # msum = np.sum(mat, axis=2)
    # msum[msum == 0] = 1
    # mat = mat / msum[:, :, None]
    # mat = mat.reshape(-1, mat.shape[2]).T

    mat = mat.squeeze(axis=2)

    # plt.figure(figsize=(16, 9))
    # ax = plt.gca()
    # ax.matshow(mat, cmap=plt.get_cmap('jet'), vmin=0, vmax=1)
    # plt.show()

    plot_mat(mat, hds.predicates, hds.objects)
    plt.show()


def vis_masks():
    results = _setup_and_load()
    hds = HicoDetInstanceSplit.get_split(split=Splits.TEST)
    hdsl = hds.get_loader(batch_size=1, shuffle=False)

    output_dir = os.path.join('analysis', 'output', 'vis', *(cfg.program.output_path.split('/')[1:]))
    dataset_classes = hds.objects

    for b_idx, example in enumerate(hdsl):
        example = example  # type: Minibatch
        prediction_dict = results[b_idx]
        prediction = Prediction.from_dict(prediction_dict)

        boxes = prediction.obj_boxes
        obj_scores = prediction.obj_scores
        box_classes = np.argmax(obj_scores, axis=1)
        box_class_scores = obj_scores[np.arange(boxes.shape[0]), box_classes]
        print(boxes)
        print(box_classes)
        print(box_class_scores)

        boxes_with_scores = np.concatenate([boxes, box_class_scores[:, None]], axis=1)
        im_fn = example.other_ex_data[0]['fn']
        im = cv2.imread(os.path.join(hds.img_dir, im_fn))
        vis_one_image(
            im[:, :, [2, 1, 0]],  # BGR -> RGB for visualization
            boxes=boxes_with_scores,
            box_classes=box_classes,
            class_names=dataset_classes,
            masks=None,
            output_file_path=os.path.join(output_dir, os.path.splitext(im_fn)[0]),
            box_alpha=0.3,
            show_class=True,
            thresh=0.0,  # Lower this to see all the predictions (was 0.7 in the original code)
            ext='png'
        )
        break


def main():
    funcs = {'vis': vis_masks,
             'att': att,
             'stats': stats,
             }

    sys.argv[1:] = ['stats', '--load_precomputed_feats']
    parser = argparse.ArgumentParser()
    parser.add_argument('func', type=str, choices=funcs.keys())
    namespace = parser.parse_known_args()
    func = vars(namespace[0])['func']
    sys.argv = sys.argv[:1] + namespace[1]
    print(sys.argv)
    funcs[func]()


if __name__ == '__main__':
    main()
