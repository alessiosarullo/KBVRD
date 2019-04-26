import numpy as np
import torch

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec

from analysis.utils import plot_mat
from config import cfg
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.dataset.utils import Splits, Example
from lib.knowledge_extractors.imsitu_knowledge_extractor import ImSituKnowledgeExtractor
from lib.models.utils import Prediction
from lib.stats.evaluator import Evaluator
from lib.dataset.utils import get_counts

matplotlib.use('Qt5Agg')

def main():
    np.random.seed(3)
    import sys
    # sys.argv += ['--prinds', '2,3,5,6,7,8,9,10,11,13,15,16,17,19,22,23,24,25,27,28,29,33,36,40,41,'
    #                          '43,45,48,49,51,52,53,55,56,58,59,60,61,63,64,66,67,69,70,71,73,74,75,'
    #                          '76,77,80,81,83,84,88,89,90,91,102,104,105,107,108,111']
    cfg.parse_args(allow_required=False)
    # np.set_printoptions(precision=2, suppress=True, linewidth=300, edgeitems=20)

    imsitu_ke = ImSituKnowledgeExtractor()
    hd = HicoDetInstanceSplit.get_split(Splits.TRAIN)  # type: HicoDetInstanceSplit

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

    predicates = hd.predicates
    objects = hd.objects
    # pred_intersection_binmask = np.any(imsitu_prior, axis=0)
    # pred_intersection_binmask[0] = True  # do not discard __no_interaction__
    # pred_intersection = np.flatnonzero(pred_intersection_binmask)
    # assert np.all(pred_intersection_binmask), np.flatnonzero(~pred_intersection_binmask)
    # # print(','.join([str(i) for i in pred_intersection]))
    # predicates = [predicates[i] for i in pred_intersection]
    # imsitu_prior = imsitu_prior[:, pred_intersection]
    # freq_prior = freq_prior[:, pred_intersection]

    hd = HicoDetInstanceSplit.get_split(Splits.TEST)  # type: HicoDetInstanceSplit
    if 0:
        all_predictions = []
        modes = ['freq', 'imsitu', 'sum', 'smax', 'mul', 'gt']
        mode = modes[-2]
        print('Mode: %s.' % mode)
        for im_idx in range(len(hd)):
            ex = hd.get_entry(im_idx, read_img=False, ignore_precomputed=True)  # type: Example
            obj_labels = ex.gt_obj_classes[ex.gt_hois[:, 2]]

            if mode == modes[0]:
                hoi_scores = freq_prior[obj_labels, :].astype(np.float)
            elif mode == modes[1]:
                hoi_scores = imsitu_prior[obj_labels, :].astype(np.float)
            elif mode == modes[2]:
                hoi_scores = prior_sum[obj_labels, :].astype(np.float)
            elif mode == modes[3]:
                hoi_scores = prior_smax[obj_labels, :].astype(np.float)
            elif mode == modes[4]:
                hoi_scores = prior_mul[obj_labels, :].astype(np.float)
            elif mode == modes[-1]:
                hoi_labels_onehot = np.zeros((ex.gt_hois.shape[0], hd.num_predicates))
                hoi_labels_onehot[np.arange(hoi_labels_onehot.shape[0]), ex.gt_hois[:, 1]] = 1
                hoi_scores = hoi_labels_onehot
            else:
                raise ValueError('Unknown mode %s.' % mode)

            prediction = Prediction(obj_im_inds=np.array([im_idx]),
                                    obj_boxes=ex.gt_boxes,
                                    obj_scores=[],
                                    ho_img_inds=np.array([im_idx]),
                                    ho_pairs=ex.gt_hois[:, [0, 2]],
                                    action_scores=hoi_scores)
            all_predictions.append(vars(prediction))
            if im_idx % 20 == 0:
                torch.cuda.empty_cache()  # Otherwise after some epochs the GPU goes out of memory. Seems to be a bug in PyTorch 0.4.1.

        evaluator = Evaluator.evaluate_predictions(hd, all_predictions, iou_thresh=0.999)  # type: Evaluator
        evaluator.print_metrics()

        for i, p in enumerate(hd.predicates):
            print('%3d %s' % (i, p))
    else:
        # Plot
        # objects = ['hair_drier']
        # idx = hd.objects.index('hair_drier')
        # freq_prior = freq_prior[[idx], :]
        # imsitu_prior = imsitu_prior[[idx], :]
        # plot_mat((imsitu_prior + freq_prior * 2) / 3, predicates, objects)

        # plt.figure(figsize=(16, 9))
        # gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1],
        #                        wspace=0.01, hspace=0.4, top=0.9, bottom=0.1, left=0.05, right=0.95)
        #
        # plot_mat(freq_prior, predicates, objects, axes=plt.subplot(gs[0, 0]))
        # plot_mat(imsitu_prior, predicates, objects, axes=plt.subplot(gs[0, 1]))
        #
        # plot_mat(prior_mul, predicates, objects, axes=plt.subplot(gs[1, 0]))
        #
        # plot_mat(prior_sum, predicates, objects, axes=plt.subplot(gs[1, 1]))
        # plt.show()

        plot_mat(imsitu_prior, predicates, objects)
        known_mask = known_objects[:, None] & known_predicates[None, :]
        plot_mat(known_mask.astype(np.float), predicates, objects)
        plt.show()


if __name__ == '__main__':
    main()
