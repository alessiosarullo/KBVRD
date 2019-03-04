import os
import pickle
import random

import numpy as np
from drivers.conceptnet_driver import Conceptnet
from drivers.hicodet_driver import HicoDet
from analysis.utils import plot_mat

from lib.dataset.word_embeddings import WordEmbeddings


#TODO class
# TODO check after Hico predicate refactor

def print_edges(edges):
    print('\n'.join([' '.join(['%40s' % str(entry) for entry in e.values()]) for e in edges]))


def tailor_to(cnet, len_max_path, dataset='HICO-DET',):
    assert dataset in ['HICO-DET']

    if dataset == 'HICO-DET':
        hd = HicoDet()
        predicates = hd.predicates
        objects = hd.objects
        print(predicates)
        print(objects)

        # predicates = ['blow']
        # objects = random.sample(objects, 20)

        # Compute paths
        try:
            with open(os.path.join('cache', 'paths%d.pkl' % len_max_path), 'rb') as f:
                all_paths = pickle.load(f)
        except FileNotFoundError:
            all_paths = [cnet.find_paths(pred, objects, max_length=len_max_path) for pred in predicates]
            with open(os.path.join('cache', 'paths%d.pkl' % len_max_path), 'wb') as f:
                pickle.dump(all_paths, f)

        wes = [WordEmbeddings(source='numberbatch'),
               WordEmbeddings(source='glove')]

        # Print paths
        hico_we_sim_mats = []
        for we in wes:
            hico_preds_we = np.atleast_2d(np.array([we.embedding(p) for p in predicates]))
            hico_objs_we = np.array([we.embedding(obj.replace('_', ' ')) for obj in objects])
            hico_we_sim = hico_objs_we.dot(hico_preds_we.T)
            hico_we_sim_mats.append(hico_we_sim)
        hico_we_sim_mat = np.stack(hico_we_sim_mats, axis=2)
        print_lines = []
        for pred, pred_paths in zip(predicates, all_paths):
            # if not pred == 'wash':
            #     continue
            print_lines += ['#' * 50 + ' ' + pred]
            pi = predicates.index(pred)
            occurrences = {obj: hd.get_occurrences([pred, obj]) for obj in objects}
            for obj in sorted(objects, key=lambda x: occurrences[x], reverse=True):
                oi = objects.index(obj)
                print_lines += ['%-20s > #occurrences: %2d. Similarity: %s' % (obj, occurrences[obj],
                                                                               ' '.join(['%6.3f' % x for x in hico_we_sim_mat[oi, pi, :]]),
                                                                               )]
                for path in pred_paths.get_split(obj, []):
                    print_lines += [' '.join([('%-20s' if i % 2 == 0 and i > 0 else '%20s') % str(x)
                                              for i, x in enumerate(path.split(cnet.PATH_SEP))])]
        with open(os.path.join('cache', 'paths%d.txt' % len_max_path), 'w', encoding='utf-8') as f:
            paths_str = '\n'.join(print_lines)
            f.write(paths_str)

        # Conceptnet object-predicate matrix
        cnet_op_mat = np.array([[1 if obj in pred_paths.keys() else 0 for obj in objects] for pred_paths in all_paths]).T

        # Hico object-predicate matrix
        pred_to_idx = {k: i for i, k in enumerate(hd.predicate_dict.keys())}
        obj_to_idx = {o: i for i, o in enumerate(hd.objects)}
        hico_op_mat = np.zeros([len(obj_to_idx), len(pred_to_idx)])
        for inter in hd.interactions:
            hico_op_mat[obj_to_idx[inter['obj']], pred_to_idx[inter['pred']]] = 1

        # Plot
        plot_mat((cnet_op_mat + hico_op_mat * 2) / 3, predicates, objects)
    else:
        raise ValueError('Unknown dataset')


def main():
    random.seed(3)

    cnet = Conceptnet()
    tailor_to(cnet, len_max_path=2, dataset='HICO-DET')
    # for k, v in cnet.get_rel_occurrences().items():
    #     print('%7d %s' % (v, k))

    # print_edges([cnet[i] for i in random.sample(range(len(cnet)), 10)])
    # print()
    # print(cnet.hico_rels)

    # print_edges(cnet.find_relations_on(query='wash', pos=None, include_unk_pos=False))


if __name__ == '__main__':
    main()
