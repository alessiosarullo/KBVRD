import numpy as np
from analysis.utils import plot_mat

from lib.dataset.hicodet_driver import HicoDet


def main():
    hd = HicoDet()

    pred_to_idx = {k: i for i, k in enumerate(hd.predicate_dict.keys())}
    obj_to_idx = {o: i for i, o in enumerate(hd.objects)}

    op_mat = np.zeros([len(obj_to_idx), len(pred_to_idx)])
    for inter in hd.interactions:
        op_mat[obj_to_idx[inter['obj']], pred_to_idx[inter['pred']]] = 1
    pred_labels = hd.predicates
    obj_labels = hd.objects

    # # Sort by most frequent object and predicate
    # num_objs_per_predicate = np.sum(op_mat, axis=0)
    # inds = np.argsort(num_objs_per_predicate)[::-1]
    # pred_labels = [pred_labels[i] for i in inds]
    # op_mat = op_mat[:, inds]
    #
    # num_preds_per_object = np.sum(op_mat, axis=1)
    # inds = np.argsort(num_preds_per_object)[::-1]
    # obj_labels = [obj_labels[i] for i in inds]
    # op_mat = op_mat[inds, :]
    #
    # # Use different colors
    # for i in range(op_mat.shape[0]):
    #     for j in range(op_mat.shape[1]):
    #         op_mat[i, j] -= 0.5 * i / op_mat.shape[0]
    #         op_mat[i, j] -= 0.5 * j / op_mat.shape[1]

    plot_mat(op_mat, pred_labels, obj_labels)


if __name__ == '__main__':
    main()
