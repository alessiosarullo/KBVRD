import argparse
import os
import pickle
import sys
from typing import Dict, List

import matplotlib
import nltk
import numpy as np
from matplotlib import pyplot as plt
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset, Lemma

from analysis.utils import plot_mat
from config import cfg
from lib.dataset.hicodet.hicodet_split import HicoDetSplitBuilder, HicoDetSplit, Splits, HicoDet
from lib.dataset.hicodet.pc_hicodet_split import PrecomputedHicoDetSplit
from lib.dataset.word_embeddings import WordEmbeddings

try:
    matplotlib.use('Qt5Agg')
    # sys.argv[1:] = ['embop']
    sys.argv[1:] = ['embpp']
    # sys.argv[1:] = ['hois']
except ImportError:
    pass


def most_similar(sim_mat, classes_rows, classes_cols=None, filter_first=True, top=11):
    if classes_cols is None:
        classes_cols = classes_rows
    sorted_by_sim = sim_mat.argsort(axis=1)[:, ::-1]
    s = 1 if filter_first else 0
    for i in range(sim_mat.shape[0]):
        print('%20s: %s' % (classes_rows[i], ','.join(['%20s (%.3f)' % (classes_cols[j], sim_mat[i, j])
                                                       for j in sorted_by_sim[i, s:top] if sim_mat[i, j] > 0])))
    print()


def plot_embedding_op_sim():
    cfg.parse_args(fail_if_missing=False, reset=True)
    dataset = HicoDet()

    # op_mat = np.zeros([dataset.num_object_classes, dataset.num_predicates])
    # for p, o in dataset.interactions:
    #     op_mat[o, p] = 1
    # plot_mat(op_mat, dataset.predicates, dataset.objects, plot=False)

    op_sims = []

    word_emb_dim = 300
    word_embs = WordEmbeddings(source='glove', dim=word_emb_dim)
    oe = word_embs.get_embeddings(dataset.objects, retry='avg_norm_last')
    pe = word_embs.get_embeddings(dataset.predicates, retry='avg_norm_first')
    oe /= np.linalg.norm(oe, axis=1, keepdims=True) + 1e-6
    pe /= np.linalg.norm(pe, axis=1, keepdims=True) + 1e-6
    op_sim = np.sum(oe[:, None, :] * pe[None, :, :], axis=2)
    # op_sim_exp = np.exp(5 * op_sim)
    # op_sim = np.maximum(op_sim_exp / op_sim_exp.sum(axis=1, keepdims=True), op_sim_exp / op_sim_exp.sum(axis=0, keepdims=True))
    # op_sim = (op_sim - op_sim.min()) / (op_sim.max() - op_sim.min())
    op_sims.append(op_sim)
    op_sim[np.logical_or(op_sim > 0.99, op_sim < 0.2)] = 0
    most_similar(op_sim, dataset.objects, dataset.predicates, filter_first=False)
    plot_mat(op_sim, dataset.predicates, dataset.objects, plot=False)

    # emb_dim = 1000
    # emb_range = (24.0 + 2.0) / emb_dim  # (self.gamma.item() + self.epsilon) / hidden_dim
    # PI = 3.14159265358979323846
    # entity_embs = np.load('cache/rotate/entity_embedding.npy')
    # with open('cache/rotate/entities.dict', 'r') as f:
    #     ecl_idx, entity_classes = zip(*[l.strip().split('\t') for l in f.readlines()])  # the index is loaded just for assertion check.
    #     ecl_idx = [int(x) for x in ecl_idx]
    #     assert np.all(np.arange(len(ecl_idx)) == np.array(ecl_idx))
    #     entity_inv_index = {e: i for i, e in enumerate(entity_classes)}
    # rotrel_phase_embs = np.load('cache/rotate/relation_embedding.npy')
    # rotrel_embs = rotrel_phase_embs / (emb_range / PI)
    # re_rotrel_embs = np.cos(rotrel_embs)
    # im_rotrel_embs = np.sin(rotrel_embs)
    # with open('cache/rotate/relations.dict', 'r') as f:
    #     rcl_idx, rot_relation_classes = zip(*[l.strip().split('\t') for l in f.readlines()])  # the index is loaded just for assertion check.
    #     rcl_idx = [int(x) for x in rcl_idx]
    #     assert np.all(np.arange(len(rcl_idx)) == np.array(rcl_idx))
    #     rotrel_inv_index = {r: i for i, r in enumerate(rot_relation_classes)}
    # oe = entity_embs[np.array([entity_inv_index[o] for o in dataset.objects])]
    # pe = np.concatenate([np.zeros((1, entity_embs.shape[1])),
    #                      entity_embs[np.array([entity_inv_index[p] for p in dataset.get_preds_for_embs()[1:]])]
    #                      ], axis=0)
    # rot_op_sims = np.zeros((dataset.num_object_classes, dataset.num_predicates, rotrel_embs.shape[0]))
    # re_pred, im_pred = pe[:, :emb_dim], pe[:, emb_dim:]
    # re_obj, im_obj = oe[:, :emb_dim][:, None, :], oe[:, emb_dim:][:, None, :]
    # for i in range(rotrel_embs.shape[0]):
    #     re_dist = (re_pred * re_rotrel_embs[None, i] - im_pred * im_rotrel_embs[None, i])[None, :, :] - re_obj
    #     im_dist = (re_pred * im_rotrel_embs[None, i] + im_pred * re_rotrel_embs[None, i])[None, :, :] - im_obj
    #     dist = np.linalg.norm(np.linalg.norm(np.stack([re_dist, im_dist], axis=3), ord=2, axis=3), ord=1, axis=2)
    #     rot_op_sims[:, :, i] = -dist
    # op_sim = rot_op_sims.max(axis=2)
    # plot_mat(op_sim, dataset.predicates, dataset.objects, plot=False, vrange=None)
    # # op_sim = (op_sim - op_sim.min()) / (op_sim.max() - op_sim.min())
    # # op_sim_exp = np.exp(5 * op_sim)
    # # op_sim = np.maximum(op_sim_exp / op_sim_exp.sum(axis=1, keepdims=True), op_sim_exp / op_sim_exp.sum(axis=0, keepdims=True))
    # # op_sim = (op_sim - op_sim.min()) / (op_sim.max() - op_sim.min())
    # # op_sim[:, 0] = 0
    # # op_sims.append(op_sim)
    # # plot_mat(op_sim, dataset.predicates, dataset.objects, plot=False)

    plt.show()


def get_hd_co_occurrences():
    hdpc = HicoDetSplitBuilder.get_splits(PrecomputedHicoDetSplit, Splits.TRAIN)  # type: PrecomputedHicoDetSplit
    co_occurrences = np.zeros([hdpc.num_actions, hdpc.num_actions])
    act_labels_bool = hdpc.pc_action_labels.astype(np.bool)
    for i in range(hdpc.num_actions):
        i_mask = act_labels_bool[:, i]
        for j in range(i + 1, hdpc.num_actions):
            j_mask = act_labels_bool[:, j]
            co_occurrences[i, j] = co_occurrences[j, i] = np.sum(i_mask & j_mask)
    return co_occurrences


def plot_embedding_pp_sim():
    cfg.parse_args(fail_if_missing=False, reset=True)
    dataset = HicoDetSplitBuilder.get_split(HicoDetSplit, split=Splits.TRAIN)

    from tools.verb_graph import get_verb_similarity, get_synonyms

    # # Co-occurrences
    # co_occurrences = get_hd_co_occurrences()
    # np.save('tmp.npy', co_occurrences)
    co_occurrences = np.load('tmp.npy')
    assert np.all(co_occurrences[np.arange(dataset.num_actions), np.arange(dataset.num_actions)] == 0)
    pp_sim = co_occurrences / np.maximum(1, np.sum(co_occurrences, axis=1, keepdims=True))
    most_similar(pp_sim, dataset.predicates, filter_first=False)
    plot_mat(pp_sim, dataset.predicates, dataset.predicates, plot=False, bin_colours=True)

    # # # WordNet
    # hd = dataset.hicodet
    # synsets_per_pred = [[wn.synset(hd.driver.wn_predicate_dict[wid]['wname']) for wid in hd.driver.predicate_dict[pred]['wn_ids']]
    #                     for pred in hd.predicates]  # type: List[List[Synset]]
    # lemmas_per_synset_per_pred = [[synset.lemma_names() for synset in synsets] for synsets in synsets_per_pred]  # type: List[List[Lemma]]
    # # Check
    # for i, synsets in enumerate(synsets_per_pred):
    #     defs = [hd.driver.wn_predicate_dict[wid]['def'] for wid in hd.driver.predicate_dict[hd.predicates[i]]['wn_ids']]
    #     wndefs = [s.definition() for s in synsets]
    #     assert len(wndefs) == len(defs) and all([d == wnd for d, wnd in zip(defs, wndefs)])
    # print()
    # for i, synsets in enumerate(synsets_per_pred):
    #     print('%3d%1s %20s: ' % (i, '*' if len(lemmas_per_synset_per_pred[i]) > 1 else '', hd.predicates[i]), lemmas_per_synset_per_pred[i])
    #     print('%3s%1s %20s  ' % ('', '', ''), [synset.definition() for synset in synsets])
    # # Actual similarity
    # sim_mat = get_verb_similarity(synsets_per_pred, threshold=False)
    # syn_sim_mat = get_synonyms(synsets_per_pred)
    # most_similar(sim_mat, hd.predicates, filter_first=False)
    # # plot_mat(sim_mat, hd.predicates, hd.predicates, vrange=None, plot=False)

    # # GloVe word embeddings
    with open(os.path.join(cfg.cache_root, 'glove_300_norm_pred-avg.pkl'), 'rb') as f:
        pe = pickle.load(f)
    pp_sim = np.sum(pe[:, None, :] * pe[None, :, :], axis=2)
    most_similar(pp_sim, dataset.predicates)
    plot_mat(pp_sim, dataset.predicates, dataset.predicates, plot=False, bin_colours=True)

    # # # ConceptNet RotatE embeddings
    # emb_dim = 1000
    # emb_range = (24.0 + 2.0) / emb_dim  # (self.gamma.item() + self.epsilon) / hidden_dim
    # PI = 3.14159265358979323846
    # entity_embs = np.load('cache/rotate/entity_embedding.npy')
    # with open('cache/rotate/entities.dict', 'r') as f:
    #     ecl_idx, entity_classes = zip(*[l.strip().split('\t') for l in f.readlines()])  # the index is loaded just for assertion check.
    #     ecl_idx = [int(x) for x in ecl_idx]
    #     assert np.all(np.arange(len(ecl_idx)) == np.array(ecl_idx))
    #     entity_inv_index = {e: i for i, e in enumerate(entity_classes)}
    # rotrel_phase_embs = np.load('cache/rotate/relation_embedding.npy')
    # rotrel_embs = rotrel_phase_embs / (emb_range / PI)
    # re_rotrel_embs = np.cos(rotrel_embs)
    # im_rotrel_embs = np.sin(rotrel_embs)
    # with open('cache/rotate/relations.dict', 'r') as f:
    #     rcl_idx, rot_relation_classes = zip(*[l.strip().split('\t') for l in f.readlines()])  # the index is loaded just for assertion check.
    #     rcl_idx = [int(x) for x in rcl_idx]
    #     assert np.all(np.arange(len(rcl_idx)) == np.array(rcl_idx))
    # pe = np.concatenate([np.zeros((1, entity_embs.shape[1])),
    #                      entity_embs[np.array([entity_inv_index[p] for p in dataset.get_preds_for_embs()[1:]])]
    #                      ], axis=0)
    # rot_pp_sims = np.zeros((dataset.num_predicates, dataset.num_predicates, rotrel_embs.shape[0]))
    # re_pred, im_pred = pe[:, :emb_dim], pe[:, emb_dim:]
    # re_pred_tail, im_pred_tail = re_pred[:, None, :], im_pred[:, None, :]
    # for i in range(rotrel_embs.shape[0]):
    #     re_dist = (re_pred * re_rotrel_embs[None, i] - im_pred * im_rotrel_embs[None, i])[None, :, :] - re_pred_tail
    #     im_dist = (re_pred * im_rotrel_embs[None, i] + im_pred * re_rotrel_embs[None, i])[None, :, :] - im_pred_tail
    #     dist = np.linalg.norm(np.linalg.norm(np.stack([re_dist, im_dist], axis=3), ord=2, axis=3), ord=1, axis=2)
    #     rot_pp_sims[:, :, i] = -dist
    # pp_sim = rot_pp_sims.max(axis=2)
    # pp_sim[:, 0] = np.min(pp_sim)
    # pp_sim[0, :] = np.min(pp_sim)
    # pp_sim = (pp_sim - pp_sim.min()) / (pp_sim.max() - pp_sim.min())
    # most_similar(pp_sim, dataset.predicates)
    # # plot_mat(pp_sim, dataset.predicates, dataset.predicates, plot=False, vrange=None, bin_colours=True)

    plt.show()


def plot_feasible_hois():
    cfg.parse_args(fail_if_missing=False, reset=True)
    dataset = HicoDet()

    hico_op_mat = np.zeros([dataset.num_objects, dataset.num_actions])
    for p, o in dataset.interactions:
        hico_op_mat[o, p] = 1

    op_mats = []

    # imSitu
    from lib.dataset.imsitu import ImSituKnowledgeExtractor
    imsitu_op_mat = (ImSituKnowledgeExtractor().extract_freq_matrix(dataset) > 0).astype(np.float)
    plot_mat((imsitu_op_mat + hico_op_mat * 2) / 3, dataset.predicates, dataset.objects, plot=False, vrange=None)
    op_mats.append(imsitu_op_mat)

    # VG-SGG
    from lib.dataset.vgsgg_driver import VGSGG
    vgsgg_op_mat = (VGSGG().get_hoi_freq(dataset) > 0).astype(np.float)
    plot_mat((vgsgg_op_mat + hico_op_mat * 2) / 3, dataset.predicates, dataset.objects, plot=False)
    op_mats.append(vgsgg_op_mat)

    # HCVRD
    from lib.dataset.hcvrd_driver import HCVRD
    hcvrd_op_mat = (HCVRD().get_hoi_freq(dataset) > 0).astype(np.float)
    plot_mat((hcvrd_op_mat + hico_op_mat * 2) / 3, dataset.predicates, dataset.objects, plot=False)
    op_mats.append(hcvrd_op_mat)

    # VG
    with open(os.path.join(cfg.cache_root, 'vg_predicate_objects.pkl'), 'rb') as f:
        vg_po = pickle.load(f)  # type: Dict[str, List[str]]
    vg_hd_po = {pred: [o for o in vg_objs if o in dataset.objects] for pred, vg_objs in vg_po.items()}
    vg_op_mat = np.zeros((dataset.num_objects, dataset.num_actions))
    for pred, objs in vg_hd_po.items():
        pi = dataset.predicate_index[pred]
        for obj in objs:
            oi = dataset.object_index[obj]
            vg_op_mat[oi, pi] = 1
    op_mats.append(vg_op_mat)

    all_op_mat = (sum(op_mats) > 0).astype(np.float)
    summary_mat = (all_op_mat + hico_op_mat * 2) / 3
    fn = np.sum(summary_mat[:, 1:] == 2 / 3)
    tp = np.sum(summary_mat[:, 1:] == 1)
    print('Missing:', fn, fn + tp, fn / (fn + tp))
    plot_mat(summary_mat, dataset.predicates, dataset.objects, plot=False)

    plt.show()


def main():
    funcs = {
        'embop': plot_embedding_op_sim,
        'embpp': plot_embedding_pp_sim,
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
