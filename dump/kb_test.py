import matplotlib
import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt

from analysis.utils import plot_mat
from config import cfg
from lib.dataset.hicodet.hicodet_hoi_split import HicoDetSplitBuilder, HicoDetHoiSplit, Example
from lib.dataset.utils import Splits
from lib.dataset.utils import get_counts
from dump.dataset.imsitu import ImSituKnowledgeExtractor
from lib.models.containers import Prediction
from lib.stats.evaluator import Evaluator
from lib.stats.utils import MetricFormatter

matplotlib.use('Qt5Agg')


def main():
    np.random.seed(3)
    # sys.argv += ['--prinds', '2,3,5,6,7,8,9,10,11,13,15,16,17,19,22,23,24,25,27,28,29,33,36,40,41,'
    #                          '43,45,48,49,51,52,53,55,56,58,59,60,61,63,64,66,67,69,70,71,73,74,75,'
    #                          '76,77,80,81,83,84,88,89,90,91,102,104,105,107,108,111']
    cfg.parse_args(fail_if_missing=False)
    # np.set_printoptions(precision=2, suppress=True, linewidth=300, edgeitems=20)

    imsitu_ke = ImSituKnowledgeExtractor()
    hd = HicoDetSplitBuilder.get_split(HicoDetHoiSplit, Splits.TRAIN)  # type: HicoDetHoiSplit

    imsitu_op_mat, known_objects, known_predicates = imsitu_ke.extract_freq_matrix(hd, return_known_mask=True)
    imsitu_prior = imsitu_op_mat.copy()
    # imsitu_prior = (imsitu_prior > 0).astype(imsitu_prior.dtype)
    imsitu_prior[:, 0] = 0
    imsitu_prior[~np.any(imsitu_prior, axis=1), 1:] = 1
    imsitu_prior /= np.sum(imsitu_prior, axis=1, keepdims=True)
    assert not np.any(np.isinf(imsitu_prior)) and not np.any(np.isnan(imsitu_prior)) and np.allclose(np.sum(imsitu_prior, axis=1), 1)

    # Hico object-predicate matrix
    freq_op_mat = get_counts(hd)
    freq_prior = freq_op_mat.astype(np.float)
    freq_prior[:, 0] = 0
    freq_prior[~np.any(freq_prior, axis=1), 1:] = 1
    freq_prior /= np.sum(freq_prior, axis=1, keepdims=True)
    assert not np.any(np.isinf(freq_prior)) and not np.any(np.isnan(freq_prior)) and np.allclose(np.sum(freq_prior, axis=1), 1)

    prior_sum = freq_op_mat + imsitu_op_mat
    prior_sum[~np.any(prior_sum, axis=1), :] = 1
    prior_sum /= np.sum(prior_sum, axis=1, keepdims=True)

    prior_smax = np.maximum(freq_prior, imsitu_prior)
    prior_smax = np.exp(10 * prior_smax)
    prior_smax[:, 0] = 0
    prior_smax /= np.sum(prior_smax, axis=1, keepdims=True)

    prior_mul = (freq_prior + 1e-6) * (imsitu_prior + 1e-6)
    prior_mul[:, 0] = 0
    prior_mul /= np.sum(prior_mul, axis=1, keepdims=True)

    predicates = hd.actions
    objects = hd.objects
    # pred_intersection_binmask = np.any(imsitu_prior, axis=0)
    # pred_intersection_binmask[0] = True  # do not discard __no_interaction__
    # pred_intersection = np.flatnonzero(pred_intersection_binmask)
    # assert np.all(pred_intersection_binmask), np.flatnonzero(~pred_intersection_binmask)
    # # print(','.join([str(i) for i in pred_intersection]))
    # actions = [actions[i] for i in pred_intersection]
    # imsitu_prior = imsitu_prior[:, pred_intersection]
    # freq_prior = freq_prior[:, pred_intersection]

    hd = HicoDetSplitBuilder.get_split(HicoDetHoiSplit, Splits.TEST)  # type: HicoDetHoiSplit
    if 1:
        possible_modes = ['freq', 'imsitu',
                          'sum', 'smax', 'mul',
                          'rnd', 'gt']
        evaluator = Evaluator(hd, iou_thresh=0.999)  # type: Evaluator
        results = []

        for m in [0, 1, 4, 5]:
            all_predictions = []
            mode = possible_modes[m]
            print('Mode: %s.' % mode)
            for im_idx in range(len(hd)):
                ex = hd.get_img_entry(im_idx, read_img=False)  # type: Example
                obj_labels_onehot = np.zeros((ex.gt_obj_classes.shape[0], hd.num_objects))
                obj_labels_onehot[np.arange(obj_labels_onehot.shape[0]), ex.gt_obj_classes] = 1
                hoi_obj_labels = ex.gt_obj_classes[ex.gt_hois[:, 2]]

                if mode == possible_modes[0]:
                    action_scores = freq_prior[hoi_obj_labels, :].astype(np.float)
                elif mode == possible_modes[1]:
                    action_scores = imsitu_prior[hoi_obj_labels, :].astype(np.float)
                elif mode == possible_modes[2]:
                    action_scores = prior_sum[hoi_obj_labels, :].astype(np.float)
                elif mode == possible_modes[3]:
                    action_scores = prior_smax[hoi_obj_labels, :].astype(np.float)
                elif mode == possible_modes[4]:
                    action_scores = prior_mul[hoi_obj_labels, :].astype(np.float)
                elif mode == possible_modes[-2]:
                    hoi_labels_onehot = np.zeros((ex.gt_hois.shape[0], hd.num_actions))
                    hoi_labels_onehot[
                        np.arange(hoi_labels_onehot.shape[0]), np.random.randint(hd.num_actions, size=hoi_labels_onehot.shape[0])] = 1
                    action_scores = hoi_labels_onehot
                elif mode == possible_modes[-1]:
                    hoi_labels_onehot = np.zeros((ex.gt_hois.shape[0], hd.num_actions))
                    hoi_labels_onehot[np.arange(hoi_labels_onehot.shape[0]), ex.gt_hois[:, 1]] = 1
                    action_scores = hoi_labels_onehot
                else:
                    raise ValueError('Unknown mode %s.' % mode)

                prediction = Prediction(obj_im_inds=np.full_like(ex.gt_obj_classes, fill_value=im_idx),
                                        obj_boxes=ex.gt_boxes,
                                        obj_scores=obj_labels_onehot,
                                        ho_img_inds=np.array([im_idx]),
                                        ho_pairs=ex.gt_hois[:, [0, 2]],
                                        action_scores=action_scores)
                all_predictions.append(vars(prediction))

            print('Predictions computed.')
            evaluator.evaluate_predictions(all_predictions)
            evaluator.output_metrics()

            results.append(evaluator.metrics['M-mAP'])

        r = results[2] - results[0]
        mf = MetricFormatter()
        lines = []
        hoi_triplets = hd.hoi_triplets
        hois = hd.op_pair_to_interaction[hoi_triplets[:, 2], hoi_triplets[:, 1]]
        hoi_metrics = {'Diff-M-mAP': r}
        lines += mf.format_metric_and_gt_lines(hois, metrics=hoi_metrics, gt_str='GT HOIs', sort=False)
        print('\n'.join(lines))
        increment_inds = np.flatnonzero(r)
        increment = r[increment_inds]
        inds = np.argsort(increment)[::-1]
        increment_inds = increment_inds[inds]
        np.set_printoptions(precision=2, suppress=True, linewidth=300, edgeitems=20)
        print(np.stack([results[2][increment_inds] / (results[0][increment_inds] + 1e-8), r[increment_inds] * 100, increment_inds], axis=0))
        print(['%d (%s %s)' % (i, hd.full_dataset.interaction_list[i]['pred'], hd.full_dataset.interaction_list[i]['obj']) for i in increment_inds])

    else:
        # Plot
        objects = ['hair_dryer']
        idx = hd.objects.index('hair_dryer')
        freq_prior = freq_prior[[idx], :]
        imsitu_prior = imsitu_prior[[idx], :]
        plot_mat((imsitu_prior + freq_prior * 2) / 3, predicates, objects, plot=False)

        plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1],
                               wspace=0.01, hspace=0.4, top=0.9, bottom=0.1, left=0.05, right=0.95)

        plot_mat(freq_prior, predicates, objects, axes=plt.subplot(gs[0, 0]), plot=False)
        plot_mat(imsitu_prior, predicates, objects, axes=plt.subplot(gs[0, 1]), plot=False)

        plot_mat(prior_mul, predicates, objects, axes=plt.subplot(gs[1, 0]), plot=False)

        plot_mat(prior_sum, predicates, objects, axes=plt.subplot(gs[1, 1]), plot=False)
        plt.show()

        plot_mat(imsitu_prior, predicates, objects, plot=False)
        known_mask = known_objects[:, None] & known_predicates[None, :]
        plot_mat(known_mask.astype(np.float), predicates, objects, plot=False)
        plt.show()


if __name__ == '__main__':
    main()
