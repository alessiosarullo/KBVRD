import argparse
import sys

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from analysis.utils import plot_mat
from config import cfg
from lib.dataset.hicodet.hicodet_split import HicoDetSplit, Splits
from lib.dataset.vgsgg_driver import VGSGG
from lib.dataset.hcvrd_driver import HCVRD
from lib.dataset.imsitu_knowledge_extractor import ImSituKnowledgeExtractor

try:
    matplotlib.use('Qt5Agg')
    # sys.argv[1:] = ['emb']
    sys.argv[1:] = ['hois']
except ImportError:
    pass


def plot_embedding_sim():
    cfg.parse_args(allow_required=False, reset=True)
    dataset = HicoDetSplit.get_split(split=Splits.TRAIN)

    # op_mat = np.zeros([dataset.num_object_classes, dataset.num_predicates])
    # for p, o in dataset.interactions:
    #     op_mat[o, p] = 1
    # plot_mat(op_mat, dataset.predicates, dataset.objects, plot=False)

    op_sims = []

    # word_emb_dim = 300
    # word_embs = WordEmbeddings(source='glove', dim=word_emb_dim)
    # oe = word_embs.get_embeddings(dataset.objects)
    # pe = word_embs.get_embeddings(dataset.predicates)
    # oe /= np.linalg.norm(oe, axis=1, keepdims=True) + 1e-6
    # pe /= np.linalg.norm(pe, axis=1, keepdims=True) + 1e-6
    # op_sim = np.sum(oe[:, None, :] * pe[None, :, :], axis=2)
    # # op_sim_exp = np.exp(5 * op_sim)
    # # op_sim = np.maximum(op_sim_exp / op_sim_exp.sum(axis=1, keepdims=True), op_sim_exp / op_sim_exp.sum(axis=0, keepdims=True))
    # # op_sim = (op_sim - op_sim.min()) / (op_sim.max() - op_sim.min())
    # op_sims.append(op_sim)
    # plot_mat(op_sim, dataset.predicates, dataset.objects, plot=False)

    emb_dim = 1000
    emb_range = (24.0 + 2.0) / emb_dim  # (self.gamma.item() + self.epsilon) / hidden_dim
    PI = 3.14159265358979323846
    entity_embs = np.load('cache/rotate/entity_embedding.npy')
    with open('cache/rotate/entities.dict', 'r') as f:
        ecl_idx, entity_classes = zip(*[l.strip().split('\t') for l in f.readlines()])  # the index is loaded just for assertion check.
        ecl_idx = [int(x) for x in ecl_idx]
        assert np.all(np.arange(len(ecl_idx)) == np.array(ecl_idx))
        entity_inv_index = {e: i for i, e in enumerate(entity_classes)}
    rotrel_phase_embs = np.load('cache/rotate/relation_embedding.npy')
    rotrel_embs = rotrel_phase_embs / (emb_range / PI)
    re_rotrel_embs = np.cos(rotrel_embs)
    im_rotrel_embs = np.sin(rotrel_embs)
    with open('cache/rotate/relations.dict', 'r') as f:
        rcl_idx, rot_relation_classes = zip(*[l.strip().split('\t') for l in f.readlines()])  # the index is loaded just for assertion check.
        rcl_idx = [int(x) for x in rcl_idx]
        assert np.all(np.arange(len(rcl_idx)) == np.array(rcl_idx))
        rotrel_inv_index = {r: i for i, r in enumerate(rot_relation_classes)}
    oe = entity_embs[np.array([entity_inv_index[o] for o in dataset.objects])]
    pe = np.concatenate([np.zeros((1, entity_embs.shape[1])),
                         entity_embs[np.array([entity_inv_index[p] for p in dataset.get_preds_for_embs()[1:]])]
                         ], axis=0)
    rot_op_sims = np.zeros((dataset.num_object_classes, dataset.num_predicates, rotrel_embs.shape[0]))
    re_pred, im_pred = pe[:, :emb_dim], pe[:, emb_dim:]
    re_obj, im_obj = oe[:, :emb_dim][:, None, :], oe[:, emb_dim:][:, None, :]
    for i in range(rotrel_embs.shape[0]):
        re_dist = (re_pred * re_rotrel_embs[None, i] - im_pred * im_rotrel_embs[None, i])[None, :, :] - re_obj
        im_dist = (re_pred * im_rotrel_embs[None, i] + im_pred * re_rotrel_embs[None, i])[None, :, :] - im_obj
        dist = np.linalg.norm(np.linalg.norm(np.stack([re_dist, im_dist], axis=3), ord=2, axis=3), ord=1, axis=2)
        rot_op_sims[:, :, i] = -dist
    op_sim = rot_op_sims.max(axis=2)
    plot_mat(op_sim, dataset.predicates, dataset.objects, plot=False, vrange=None)
    # op_sim = (op_sim - op_sim.min()) / (op_sim.max() - op_sim.min())

    # op_sim_exp = np.exp(5 * op_sim)
    # op_sim = np.maximum(op_sim_exp / op_sim_exp.sum(axis=1, keepdims=True), op_sim_exp / op_sim_exp.sum(axis=0, keepdims=True))
    # op_sim = (op_sim - op_sim.min()) / (op_sim.max() - op_sim.min())
    # op_sim[:, 0] = 0
    # op_sims.append(op_sim)
    # plot_mat(op_sim, dataset.predicates, dataset.objects, plot=False)

    plt.show()


def plot_feasible_hois():
    cfg.parse_args(allow_required=False, reset=True)
    dataset = HicoDetSplit.get_split(split=Splits.TRAIN)

    hico_op_mat = np.zeros([dataset.num_object_classes, dataset.num_predicates])
    for p, o in dataset.interactions:
        hico_op_mat[o, p] = 1

    op_mats = []

    imsitu_op_mat = (ImSituKnowledgeExtractor().extract_freq_matrix(dataset) > 0).astype(np.float)
    plot_mat((imsitu_op_mat + hico_op_mat * 2) / 3, dataset.predicates, dataset.objects, plot=False, vrange=None)
    op_mats.append(imsitu_op_mat)

    vgsgg_op_mat = (VGSGG().get_hoi_freq(dataset.hicodet) > 0).astype(np.float)
    plot_mat((vgsgg_op_mat + hico_op_mat * 2) / 3, dataset.predicates, dataset.objects, plot=False)
    op_mats.append(vgsgg_op_mat)

    hcvrd_op_mat = (HCVRD().get_hoi_freq(dataset.hicodet) > 0).astype(np.float)
    plot_mat((hcvrd_op_mat + hico_op_mat * 2) / 3, dataset.predicates, dataset.objects, plot=False)
    op_mats.append(hcvrd_op_mat)

    all_op_mat = (sum(op_mats) > 0).astype(np.float)
    plot_mat((all_op_mat + hico_op_mat * 2) / 3, dataset.predicates, dataset.objects, plot=False)

    plt.show()


def main():
    funcs = {'emb': plot_embedding_sim,
             'hois': plot_feasible_hois,
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
