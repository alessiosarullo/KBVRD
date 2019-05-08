import os
import pickle
import random

import numpy as np

from analysis.utils import plot_mat
from config import cfg
from lib.dataset.conceptnet_driver import Conceptnet
from lib.dataset.hicodet_driver import HicoDet
from lib.dataset.word_embeddings import WordEmbeddings


class ConceptnetKnowledgeExtractor:
    def __init__(self):
        self._path_sep = '#'
        self._useless_rels = ['MannerOf', 'EtymologicallyRelatedTo', 'DerivedFrom', 'AtLocation',
                              'PartOf', 'IsA']

        self.cache_paths_fn_format = os.path.join(cfg.program.cache_root, 'cnet_paths_len%d.pkl')

    def extract_freq_matrix(self, dataset, len_max_path=1):
        assert isinstance(dataset, (HicoDet,))

        # Compute paths
        paths_fn = self.cache_paths_fn_format % len_max_path
        try:
            with open(paths_fn, 'rb') as f:
                all_paths = pickle.load(f)
        except FileNotFoundError:
            cnet = Conceptnet()
            all_paths = [self.find_paths(cnet, pred, dataset.objects, max_length=len_max_path) for pred in dataset.predicates]
            with open(paths_fn, 'wb') as f:
                pickle.dump(all_paths, f)

        self.print_paths(dataset, all_paths, self._path_sep)

        op_mat = np.array([[1 if obj in pred_paths.keys() else 0 for obj in dataset.objects] for pred_paths in all_paths]).T
        return op_mat

    def find_paths(self, cnet, src_node, dst_nodes, max_length=3, best_path_only=True):
        if not isinstance(dst_nodes, list):
            dst_nodes = [dst_nodes]

        paths = []
        path_inds_per_endpoint = {}
        frontier = set()
        forms = {src_node, src_node.split('/')[0] + '/v'}  # FIXME hard-coded
        for form in forms:
            paths.append([])
            path_inds_per_endpoint[form] = [len(paths) - 1]
            frontier.add(form)

        print(src_node, forms)

        explored_nodes = set()
        for d in range(max_length):
            explored_nodes.update(frontier)
            new_frontier = set()
            for node in frontier:
                edge_inds = self.find_outgoing_edges_inds(cnet, node)
                for edge_idx in edge_inds:
                    next_node = cnet.edges[edge_idx]['dst']
                    if next_node in explored_nodes:
                        continue
                    added = False
                    for j in path_inds_per_endpoint[node]:
                        path = paths[j]
                        if next_node in path:
                            continue
                        paths.append(path + [edge_idx])
                        path_inds_per_endpoint.setdefault(next_node, []).append(len(paths) - 1)
                        added = True
                    if added:
                        new_frontier.add(next_node)
            frontier = new_frontier

        path_inds_per_dst = {k: v for k, v in path_inds_per_endpoint.items() if k in dst_nodes}
        solution_paths = {}
        for obj, path_inds in path_inds_per_dst.items():
            sol_paths_per_length = {}
            for pi in path_inds:
                sol_paths_per_length.setdefault(len(paths[pi]), []).append(pi)
            sol_path_inds = sol_paths_per_length[min(sol_paths_per_length.keys())]

            for ind in sol_path_inds:
                path = paths[ind]
                if path:  # path might be empty in case of homonyms like 'train' (action) and 'train' (vehicle)
                    path_str = cnet.edges[path[0]]['src']
                    for edge_idx in path:
                        edge = cnet.edges[edge_idx]
                        path_str += '%s%s (%4.2f)%s%s' % (self._path_sep, edge['rel'], edge['weight'], self._path_sep, edge['dst'])
                    solution_paths.setdefault(obj, set()).add(path_str)

        # Filter paths
        solution_paths = {obj: [path for path in paths if all(bad_rel not in path for bad_rel in self._useless_rels)]
                          for obj, paths in solution_paths.items()}
        solution_paths = {k: v for k, v in solution_paths.items() if v}

        if best_path_only:
            best_paths = {}
            for obj, paths in solution_paths.items():
                scores = [sum([float(rel_token[:-1].split('(')[-1]) * 1 / (p + 1)
                               for p, rel_token in enumerate(path.split(self._path_sep)[1::2])])
                          for path in paths]
                best_paths[obj] = [paths[np.argmax(scores)]]
            return best_paths
        else:
            return {k: sorted(v) for k, v in solution_paths.items()}

    @staticmethod
    def find_outgoing_edges_inds(cnet, query, include_unk_pos=True):
        if not cnet.has_pos_tag(query):
            queries = ['%s/%s' % (query, pt) for pt in cnet.pos_tags]
            if include_unk_pos:
                queries += [query]
        else:
            queries = [query]

        results = []
        for q in queries:
            results.extend([i for i in cnet.edges_from.get(q, [])])
        return results

    @staticmethod
    def print_paths(hicodet: HicoDet, all_paths, sep, verbose=False):
        assert len(hicodet.predicates) == len(all_paths)
        wes = [WordEmbeddings(source='numberbatch', normalize=True), WordEmbeddings(source='glove', normalize=True, dim=200)]

        hico_we_sim_mats = []
        for we in wes:
            print('Embeddings:', we.source)
            hico_preds_we = we.get_embeddings(hicodet.predicates, retry='first')
            hico_objs_we = we.get_embeddings([obj.replace('_', ' ') for obj in hicodet.objects], retry='last')
            hico_we_sim = hico_objs_we.dot(hico_preds_we.T)
            hico_we_sim_mats.append(hico_we_sim)
        hico_we_sim_mat = np.stack(hico_we_sim_mats, axis=2)

        print_lines = []
        for pred, pred_paths in zip(hicodet.predicates, all_paths):
            # if not pred == 'wash':
            #     continue
            print_lines += ['#' * 50 + ' ' + pred]
            pi = hicodet.predicates.index(pred)
            occurrences = {obj: hicodet.get_occurrences([pred, obj]) for obj in hicodet.objects}
            for obj in sorted(hicodet.objects, key=lambda x: occurrences[x], reverse=True):
                oi = hicodet.objects.index(obj)
                if verbose or obj in pred_paths or occurrences[obj] > 0:
                    print_lines += ['%-20s > #occurrences: %4d. Similarity: %s' % (obj, occurrences[obj],
                                                                                   ' '.join(['% 6.3f' % x for x in hico_we_sim_mat[oi, pi, :]]),
                                                                                   )]
                for path in pred_paths.get(obj, []):
                    print_lines += [' '.join([('%-20s' if i % 2 == 0 and i > 0 else '%20s') % str(x)
                                              for i, x in enumerate(path.split(sep))])]
        print('\n'.join(print_lines))


def main():
    random.seed(3)

    cnet_ex = ConceptnetKnowledgeExtractor()
    dataset = HicoDet()
    cnet_op_mat = cnet_ex.extract_freq_matrix(dataset, len_max_path=2)

    # Hico object-predicate matrix
    hico_op_mat = np.zeros([len(dataset.objects), len(dataset.predicates)])
    for i in range(len(dataset.interaction_list)):
        hico_op_mat[dataset.get_object_index(i), dataset.get_predicate_index(i)] = 1
    plot_mat((cnet_op_mat + hico_op_mat * 2) / 3, dataset.predicates, dataset.objects)


def plot():
    cnet = Conceptnet(file_path='cache/cnet_hd2.pkl')
    print(cnet.node_index['settle'] in [i for i in cnet.edges_from[cnet.node_index['adjust']]])

    dataset = HicoDet()
    with open('cache/cnet_hd2_rel2.pkl', 'rb') as f:
        d = pickle.load(f)
        nodes = d['nodes']
        cnet_rel = d['rel']
        node_inv_index = {n: i for i, n in enumerate(nodes)}

    hico_op_mat = np.zeros([len(dataset.objects), len(dataset.predicates)])
    for i in range(len(dataset.interaction_list)):
        hico_op_mat[dataset.get_object_index(i), dataset.get_predicate_index(i)] = 1

    # l = len(nodes)
    # l = 40
    # plot_mat(cnet_rel[:l, :l], nodes[:l], nodes[:l])

    cnet_op_mat = np.zeros([len(dataset.objects), len(dataset.predicates)])
    for i, o in enumerate(dataset.objects):
        o = 'hair_dryer' if o == 'hair_drier' else o
        oidx = node_inv_index[o]
        for j, p in enumerate(dataset.predicates):
            if p == dataset.null_interaction:
                continue
            p = p.split('_')[0]
            pidx = node_inv_index[p]
            cnet_op_mat[i, j] = cnet_rel[oidx, pidx]
    cnet_op_mat = np.minimum(1, cnet_op_mat)

    plot_mat((cnet_op_mat + hico_op_mat * 2) / 3, dataset.predicates, dataset.objects)


if __name__ == '__main__':
    main()
    # plot()
