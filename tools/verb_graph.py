import os
import pickle
from typing import Dict, List

import nltk
import numpy as np
from matplotlib import pyplot as plt
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

from analysis.utils import plot_mat
from config import cfg
from lib.dataset.hcvrd_driver import HCVRD
from lib.dataset.hicodet.hicodet import HicoDet
from lib.dataset.hicodet.hicodet_split import HicoDetSplitBuilder, Splits
from lib.dataset.hicodet.pc_hicodet_split import PrecomputedHicoDetSplit
from lib.dataset.imsitu import ImSituKnowledgeExtractor
from lib.dataset.vgsgg_driver import VGSGG


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


def get_verb_similarity(synsets_per_pred, threshold=True):
    num_predicates = len(synsets_per_pred)

    lch_sim_mat = np.zeros((num_predicates, num_predicates))
    for i in range(num_predicates):
        for j in range(num_predicates):
            sims = [wn.lch_similarity(m1, m2) for m1 in synsets_per_pred[i] for m2 in synsets_per_pred[j]]
            if sims:
                lch_sim_mat[i, j] = max(sims)
    lch_sim_mat /= np.max(lch_sim_mat)
    if threshold:
        lch_sim_mat = (lch_sim_mat >= 0.6).astype(np.float)

    ic = nltk.corpus.wordnet_ic.ic('ic-brown.dat')  # weight_senses_equally=True ?
    lin_sim_mat = np.zeros((num_predicates, num_predicates))
    for i in range(num_predicates):
        for j in range(num_predicates):
            sims = [wn.lin_similarity(m1, m2, ic) for m1 in synsets_per_pred[i] for m2 in synsets_per_pred[j]]
            if sims:
                lin_sim_mat[i, j] = max(sims)
    if threshold:
        lin_sim_mat = (lin_sim_mat >= 0.7).astype(np.float)

    sim_mat = np.maximum(lch_sim_mat, lin_sim_mat)
    return sim_mat


def get_synonyms(synsets_per_pred: List[List[Synset]]):
    num_predicates = len(synsets_per_pred)
    synonyms = np.zeros((num_predicates, num_predicates))
    for i in range(num_predicates):
        pi_synsets = synsets_per_pred[i]
        pi_lemmas = {lemma_name for synset in pi_synsets for lemma_name in synset.lemma_names()}
        pi_defs = '; '.join([synset.definition() for synset in pi_synsets])
        for j in range(num_predicates):
            for synset in synsets_per_pred[j]:
                synset_lemmas = set(synset.lemma_names())
                if any([l in pi_defs for l in synset_lemmas]) or len(pi_lemmas & synset_lemmas) > 0:
                    synonyms[i, j] = 1
                    break
    return synonyms


def get_pp_similarity(hd: HicoDet):
    synsets_per_pred = [[wn.synset(hd.driver.wn_predicate_dict[wid]['wname']) for wid in hd.driver.predicate_dict[pred]['wn_ids']]
                        for pred in hd.predicates]  # type: List[List[Synset]]
    sim_mat = get_verb_similarity(synsets_per_pred)
    syn_sim_mat = get_synonyms(synsets_per_pred)
    return sim_mat, syn_sim_mat


def get_feasible_hois(dataset: HicoDet):
    op_mats = []

    # imSitu
    imsitu_op_mat = (ImSituKnowledgeExtractor().extract_freq_matrix(dataset) > 0).astype(np.float)
    op_mats.append(imsitu_op_mat)

    # VG-SGG
    vgsgg_op_mat = (VGSGG().get_hoi_freq(dataset) > 0).astype(np.float)
    op_mats.append(vgsgg_op_mat)

    # HCVRD
    hcvrd_op_mat = (HCVRD().get_hoi_freq(dataset) > 0).astype(np.float)
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
    return all_op_mat


def plot():
    cfg.parse_args(fail_if_missing=False)

    hd = HicoDet()
    # # co_occ = get_hd_co_occurrences()
    # # np.save('tmp.npy', co_occ)
    co_occ = np.load('tmp.npy')
    assert np.all(co_occ[np.arange(hd.num_actions), np.arange(hd.num_actions)] == 0)
    # co_occ = co_occ / np.maximum(1, np.sum(co_occ, axis=1, keepdims=True))
    co_occ = (co_occ > 0).astype(np.float)
    # plot_mat(co_occ, hd.predicates, hd.predicates, plot=False)

    # Object-predicate
    all_op_mat = get_feasible_hois(hd)
    hico_op_mat = np.zeros([hd.num_objects, hd.num_actions])
    for p, o in hd.interactions:
        hico_op_mat[o, p] = 1
    summary_op_mat = (all_op_mat + hico_op_mat * 2) / 3
    # plot_mat(summary_op_mat, hd.predicates, hd.objects, plot=False)

    # Predicate-predicate
    p2p_from_op = all_op_mat.T.dot(all_op_mat)
    p2p_from_op_norm = (p2p_from_op > 0).astype(np.float)
    summary_ppop_mat = (p2p_from_op_norm + co_occ * 2) / 3
    plot_mat(summary_ppop_mat, hd.predicates, hd.predicates, plot=False)

    sim_mat, syn_sim_mat = get_pp_similarity(hd)  # PP
    # plot_mat(sim_mat, hd.predicates, hd.predicates, plot=False)
    # plot_mat(syn_sim_mat, hd.predicates, hd.predicates, plot=False)
    # summary_pp_mat = (((sim_mat + syn_sim_mat) > 0).astype(np.float) + co_occ * 2) / 3
    summary_pp_mat = (syn_sim_mat + co_occ * 2) / 3
    plot_mat(summary_pp_mat, hd.predicates, hd.predicates, plot=False)

    plt.show()


def export_to_rotate_edge_list(output_dir, edges, nodes: List[str], relations: List[str]):
    output = []
    for h, r, t in edges:
        output.append(f'{h}\t{r}\t{t}')
    lines = '\n'.join(output)
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        f.write(lines)
    with open(os.path.join(output_dir, 'entities.dict'), 'w') as f:
        f.write('\n'.join([f'{i}\t{n}' for i, n in enumerate(nodes)]))
    with open(os.path.join(output_dir, 'relations.dict'), 'w') as f:
        f.write('\n'.join([f'{i}\t{r}' for i, r in enumerate(relations)]))


def main():
    cfg.parse_args(fail_if_missing=False)

    hd = HicoDet()

    op_mat = get_feasible_hois(hd)
    p2p_from_op = ((op_mat.T[:, None, :] * op_mat.T[None, :, :]) > 0)
    p2p_from_op_norm = p2p_from_op.any(axis=2).astype(np.float)

    sim_mat, syn_sim_mat = get_pp_similarity(hd)

    entities = hd.predicates
    relations = [f'Common_{o}' for o in hd.objects] + ['Synonym', 'Related']
    triples = []
    for i, pi in enumerate(hd.predicates):
        for j, pj in enumerate(hd.predicates):
            if i == j:
                continue

            common_objects = np.flatnonzero(p2p_from_op[i, j])
            triples += [[pi, f'Common_{hd.objects[o]}', pj] for o in common_objects]

            if syn_sim_mat[i, j] > 0:
                triples.append([pi, 'Synonym', pj])

            if sim_mat[i, j] > 0:
                triples.append([pi, 'Related', pj])

    # Check
    for p1, r, p2 in triples:
        assert p1 in entities and p2 in entities and r in relations

    export_to_rotate_edge_list('.', triples, entities, relations)


if __name__ == '__main__':
    main()
    # plot()
