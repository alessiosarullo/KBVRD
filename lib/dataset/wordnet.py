from typing import List

import nltk
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset, Lemma

from analysis.utils import plot_mat
from lib.dataset.hicodet.hicodet import HicoDet
from lib.dataset.hicodet.pc_hicodet_split import PrecomputedHicoDetSplit, HicoDetSplits, Splits


def get_hd_co_occurrences():
    hdpc = HicoDetSplits.get_splits(PrecomputedHicoDetSplit, Splits.TRAIN)  # type: PrecomputedHicoDetSplit
    co_occurrences = np.zeros([hdpc.num_predicates, hdpc.num_predicates])
    act_labels_bool = hdpc.pc_action_labels.astype(np.bool)
    for i in range(hdpc.num_predicates):
        i_mask = act_labels_bool[:, i]
        for j in range(i + 1, hdpc.num_predicates):
            j_mask = act_labels_bool[:, j]
            co_occurrences[i, j] = co_occurrences[j, i] = np.sum(i_mask & j_mask)
    return co_occurrences


def main():
    hd = HicoDet()

    # coocc = get_hd_co_occurrences()
    # np.save('tmp.npy', coocc)
    # coocc = np.load('tmp.npy')

    # synsets_per_pred = [wn.synsets(x.split('_')[0], pos=wn.VERB) for x in hd.predicates]  # type: List[List[Synset]]
    synsets_per_pred = [[wn.synset(hd.driver.wn_predicate_dict[wid]['wname']) for wid in hd.driver.predicate_dict[pred]['wn_ids']]
                        for pred in hd.predicates]  # type: List[List[Synset]]
    lemmas_per_synset_per_pred = [[synset.lemma_names() for synset in synsets] for synsets in synsets_per_pred]  # type: List[List[Lemma]]

    # Check
    for i, synsets in enumerate(synsets_per_pred):
        defs = [hd.driver.wn_predicate_dict[wid]['def'] for wid in hd.driver.predicate_dict[hd.predicates[i]]['wn_ids']]
        wndefs = [s.definition() for s in synsets]
        assert len(wndefs) == len(defs) and all([d == wnd for d, wnd in zip(defs, wndefs)])

    # i = 86
    # print('%3d %20s: ' % (i, hd.predicates[i]), synsets_per_pred[i])
    # print('%3d %20s: ' % (i, hd.predicates[i]), lemmas_per_synset_per_pred[i])
    #
    # x_per_synset_per_pred = [[synset.examples() for synset in synsets] for synsets in synsets_per_pred]
    # print('%3d %20s: ' % (i, hd.predicates[i]), x_per_synset_per_pred[i])
    # x_per_synset_per_pred = [[synset.definition() for synset in synsets] for synsets in synsets_per_pred]
    # print('%3d %20s: ' % (i, hd.predicates[i]), x_per_synset_per_pred[i])
    #
    # x_per_synset_per_pred = [[synset.usage_domains() for synset in synsets] for synsets in synsets_per_pred]
    # print('%3d %20s: ' % (i, hd.predicates[i]), x_per_synset_per_pred[i])
    # x_per_synset_per_pred = [[synset.causes() for synset in synsets] for synsets in synsets_per_pred]
    # print('%3d %20s: ' % (i, hd.predicates[i]), x_per_synset_per_pred[i])
    #
    # x_per_synset_per_pred = [{fid for j, synset in enumerate(synsets) for fid in synset.frame_ids()}
    #                          for i, synsets in enumerate(synsets_per_pred)]
    # print('%3d %20s: ' % (i, hd.predicates[i]), x_per_synset_per_pred[i])

    print()
    for i, synsets in enumerate(synsets_per_pred):
        print('%3d%1s %20s: ' % (i, '*' if len(lemmas_per_synset_per_pred[i]) > 1 else '', hd.predicates[i]), lemmas_per_synset_per_pred[i])
        print('%3s%1s %20s  ' % ('', '', ''), [synset.definition() for synset in synsets])

    ic = nltk.corpus.wordnet_ic.ic('ic-brown.dat')  # weight_senses_equally=True ?
    sim_mat = np.zeros((hd.num_predicates, hd.num_predicates))
    for i in range(hd.num_predicates):
        for j in range(hd.num_predicates):
            sims = [wn.jcn_similarity(m1, m2, ic) for m1 in synsets_per_pred[i] for m2 in synsets_per_pred[j]]
            if sims:
                sim_mat[i, j] = max(sims)
    plot_mat(sim_mat, hd.predicates, hd.predicates, plot=True)


if __name__ == '__main__':
    main()
