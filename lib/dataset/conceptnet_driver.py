import json
import os
import pickle
import scipy.sparse
import numpy as np
import re

from config import cfg


class Conceptnet:
    def __init__(self, file_path=None):
        self.pos_tags = ['n', 'v', 'a', 's', 'r']
        self.rels_to_filter = ['Antonym', 'DistinctFrom', 'NotCapableOf', 'NotDesires', 'NotHasProperty',
                               'dbpedia/capital', 'dbpedia/field', 'dbpedia/genre', 'dbpedia/genus', 'dbpedia/influencedBy',
                               'dbpedia/knownFor', 'dbpedia/language', 'dbpedia/leader', 'dbpedia/occupation', 'dbpedia/product']

        data_dir = os.path.join(cfg.program.data_root, 'ConceptNet')
        self.path_cnet = os.path.join(cfg.program.cache_root, 'conceptnet.pkl')
        self.path_raw_cnet = os.path.join(data_dir, 'raw_conceptnet.pkl')
        self.path_raw_cnet_eng = os.path.join(data_dir, 'conceptnet560_en.txt')
        self.path_raw_cnet_orig = os.path.join(data_dir, 'conceptnet560.csv')

        self.nodes = self.node_index = self.relations = self.relation_index = None
        self.edges = self.edge_rels = self.edge_weights = self.edges_from = self.edges_to = None
        if file_path is None:
            self._edge_dict = self._load()
        else:
            with open(file_path, 'rb') as f:
                d = pickle.load(f)
            self._edge_dict = d['_edge_dict']
        self._init()

    # Properties
    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def num_rels(self):
        return len(self.relations)

    @property
    def num_edges(self):
        return self.edges.shape[0]

    # Public methods
    def has_pos_tag(self, entry, tag=None):
        if len(entry) > 2 and entry[-2] == '/':
            assert entry[-1] in self.pos_tags
            if tag is None:
                return True
            else:
                return entry[-1] == tag
        return False

    def get_adjacency_matrix(self, sparse=False):
        num_nodes = self.num_nodes
        if num_nodes ** 2 >= 1e10 and not sparse:
            raise ValueError('Number of nodes is too big (%d) for a dense matrix.' % num_nodes)

        row, col = self.edges[:, 0], self.edges[:, 1]
        if sparse:
            adj = scipy.sparse.csr_matrix((np.ones(row.size), (row, col)))
        else:
            adj = np.zeros((num_nodes, num_nodes), dtype=np.float16)
            adj[row, col] = 1
        adj[np.arange(num_nodes), np.arange(num_nodes)] = 0
        return adj

    def filter_nodes(self, node_seed, radius=3):
        node_seed_str = set(['%s/%s' % (n, tag) for n in node_seed for tag in self.pos_tags]) | set(node_seed)
        node_set_str = set(self.nodes)
        keep = {self.node_index[n_str] for n_str in node_seed_str & node_set_str}
        neighs_r = keep
        for r in range(1, radius):
            neighs_r = {self.edges[e, 1] for n in neighs_r for e in self.edges_from[n]} - keep
            keep = keep | neighs_r

        keep_str = {self.nodes[k] for k in keep}
        self._edge_dict = [e for e in self._edge_dict if e['src'] in keep_str and e['dst'] in keep_str]
        self._init()

    def find_relations(self, src_nodes, walk_length=1):
        src_nodes = sorted(src_nodes)
        node_variants_index = {(n + tag): i for i, n in enumerate(src_nodes) for tag in [''] + ['/' + t for t in self.pos_tags]}

        nodes_of_interest = list(set(node_variants_index.keys()) & set(self.nodes))
        inds = np.array([self.node_index[h] for h in nodes_of_interest])

        adj = self.get_adjacency_matrix()
        rel_with_variants = np.zeros((inds.size, inds.size))
        left = adj[inds, :]  # this is equivalent to diag(inds) @ adj, but more computationally efficient
        right = adj[:, inds]
        if walk_length >= 1:
            rel_with_variants += adj[inds, :][:, inds]  # similarly, equivalent to diag(inds) @ adj @ diag(inds)
        for step in range(2, walk_length):
            # Equivalent to diag(inds) @ adj^step @ diag(inds)
            rel_with_variants += left @ right

            # The minimum is there because we don't care about how many paths are there
            if step + 1 < walk_length:
                if step % 2 == 0:
                    left = np.minimum(1, left @ adj)
                else:
                    right = np.minimum(1, adj @ right)
        rel_with_variants[np.arange(inds.size), np.arange(inds.size)] = 0
        rel_with_variants = np.minimum(1, rel_with_variants)

        num_nodes = len(src_nodes)
        src_to_inds = np.zeros((num_nodes, inds.size))
        for i, idx in enumerate(inds):
            src_to_inds[node_variants_index[self.nodes[idx]], i] = 1
        rel = np.minimum(1, src_to_inds @ rel_with_variants @ src_to_inds.T)
        rel[np.arange(num_nodes), np.arange(num_nodes)] = 0
        return rel, src_nodes

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump({'nodes': self.nodes,
                         'relations': self.relations,
                         'edges': self.edges,
                         'edge_rels': self.edge_rels,
                         'edge_weights': self.edge_weights,
                         '_edge_dict': self._edge_dict,  # FIXME this is not needed, but keeping it for not refactoring the constructor
                         }, f)

    # Iterator methods
    def __iter__(self):
        return self._edge_dict

    def __next__(self):
        return self._edge_dict.__next__()

    def __len__(self):
        return len(self._edge_dict)

    def __getitem__(self, item):
        return self._edge_dict[item]

    # Export methods
    def export_to_deepwalk_edge_list(self, relationship=None):
        node_index = {n: i for i, n in enumerate(self.nodes)}
        output = []
        for edge in self._edge_dict:
            if relationship is None or edge['rel'] == relationship:
                output.append('%d %d' % (node_index[edge['src']], node_index[edge['dst']]))
        lines = '\n'.join(output)
        with open(os.path.join(cfg.program.cache_root, 'cnet.edgelist'), 'w') as f:
            f.write(lines)

    def export_to_rotate_edge_list(self, output_dir, relationship=None):
        output = []
        for edge in self._edge_dict:
            if relationship is None or edge['rel'] == relationship:
                output.append('%s\t%s\t%s' % (edge['src'], edge['rel'], edge['dst']))
        lines = '\n'.join(output)
        with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
            f.write(lines)
        with open(os.path.join(output_dir, 'entities.dict'), 'w') as f:
            f.write('\n'.join(['%d\t%s' % (i, n) for i, n in enumerate(self.nodes)]))
        with open(os.path.join(output_dir, 'relations.dict'), 'w') as f:
            f.write('\n'.join(['%d\t%s' % (i, n) for i, n in enumerate(self.relations)]))

    # Private methods
    def _init(self):
        num_edges = len(self._edge_dict)

        self.nodes = sorted({e[k] for e in self._edge_dict for k in ['src', 'dst']})
        self.node_index = {n: i for i, n in enumerate(self.nodes)}
        self.relations = sorted(set([edge['rel'] for edge in self._edge_dict]))
        self.relation_index = {r: i for i, r in enumerate(self.relations)}

        self.edges = np.full((num_edges, 2), fill_value=-1, dtype=np.int)
        self.edge_rels = np.full(num_edges, fill_value=-1, dtype=np.int)
        self.edge_weights = np.zeros(num_edges)
        self.edges_from = {i: [] for i in range(len(self.nodes))}
        self.edges_to = {i: [] for i in range(len(self.nodes))}
        for i, e in enumerate(self._edge_dict):
            src = self.node_index[e['src']]
            dst = self.node_index[e['dst']]

            self.edges[i, :] = [src, dst]
            self.edge_rels[i] = self.relation_index[e['rel']]
            self.edge_weights[i] = e['weight']
            self.edges_from[src].append(i)
            self.edges_to[dst].append(i)

    def _load(self):
        try:
            with open(self.path_cnet, 'rb') as f:
                print('Loading the refined ConceptNet')
                edges = pickle.load(f)
        except FileNotFoundError:
            edges = self._load_raw()
            edges = self._extend_and_filter(edges)  # lowercase relationships are manually added
            with open(self.path_cnet, 'wb') as f:
                pickle.dump(edges, f)
        return edges

    def _extend_and_filter(self, edge_dict, mandatory_pos_tag=False):
        new_edges = []
        to_keep = []
        for i, e in enumerate(edge_dict):
            src, rel, dst = e['src'], e['rel'], e['dst']
            # Filter
            if rel in self.rels_to_filter or \
                    (mandatory_pos_tag and not (self.has_pos_tag(src) and self.has_pos_tag(dst))) or \
                    any([bool(re.search(r"[^a-zA-Z0-9_'\-]", (word[:-2] if self.has_pos_tag(word) else word))) for word in (src, dst)]):
                continue

            # Extend
            to_keep.append(i)
            if rel == 'IsA':
                new_edges.append({'src': dst,
                                  'rel': 'hasSubtype',
                                  'dst': src,
                                  'weight': e['weight'],
                                  })
        return [edge_dict[i] for i in to_keep] + new_edges

    def _load_raw(self):
        def _parse_concept(_raw_concept, filter_pos=False):
            assert _raw_concept[:6] == '/c/en/'
            _c = _raw_concept[6:]
            if filter_pos and self.has_pos_tag(_c):  # filter POS suffix
                _c = _c[:-2]
            return _c

        try:
            with open(self.path_raw_cnet, 'rb') as f:
                print('Loading ConceptNet')
                parsed_entries = pickle.load(f)
        except FileNotFoundError:
            print('Converting ConceptNet')
            # Only consider english entries
            try:
                with open(self.path_raw_cnet_eng, 'r', encoding='utf-8') as f:
                    entries = f.readlines()
            except FileNotFoundError:
                print('Filtering ENG entries')
                with open(self.path_raw_cnet_orig, 'r', encoding='utf-8') as f:
                    orig_entries = f.readlines()
                entries = []
                for entry in orig_entries:
                    fields = entry.split()
                    src = fields[2]
                    dst = fields[3]
                    if src.split('/')[2] == dst.split('/')[2] == 'en':
                        entries.append(entry)
                with open(self.path_raw_cnet_eng, 'w', encoding='utf-8') as f:
                    f.writelines(entries)

            parsed_entries = []
            for l in entries:
                fields = l.split()

                src = _parse_concept(fields[2])
                dst = _parse_concept(fields[3])
                assert fields[1].startswith('/r/')
                relation = fields[1][3:]

                other = ' '.join(fields[4:])
                other_dict = json.loads(other)
                weight = other_dict['weight']

                parsed_entries.append({'src': src,
                                       'rel': relation,
                                       'dst': dst,
                                       'weight': weight,
                                       })

            with open(self.path_raw_cnet, 'wb') as f:
                pickle.dump(parsed_entries, f)
        return parsed_entries


def save_cnet_hd(radius=2, walk_length=2):
    from lib.dataset.hicodet_driver import HicoDet
    hd = HicoDet()
    cnet = Conceptnet()

    hd_preds = {p.split('_')[0] for p in set(hd.predicates) - {hd.null_interaction}}
    hd_nodes = set(['hair_dryer' if obj == 'hair_drier' else obj for obj in hd.objects]) | hd_preds
    cnet.filter_nodes(hd_nodes, radius=radius)
    cnet.save(file_path='cache/cnet_hd%d.pkl' % radius)

    rel, rel_nodes = cnet.find_relations(src_nodes=hd_nodes, walk_length=walk_length)
    assert set(rel_nodes) == set(hd_nodes)
    with open('cache/cnet_hd%d_rel%d.pkl' % (radius, walk_length), 'wb') as f:
        pickle.dump({'nodes': rel_nodes,
                     'rel': rel,
                     }, f)

    return cnet, hd, rel_nodes, rel


def main():
    from lib.dataset.hicodet_driver import HicoDet
    hd = HicoDet()

    cnet = Conceptnet(file_path='cache/cnet_hd2.pkl')
    print(cnet.nodes[:5])

    # cnet.export_to_deepwalk_edge_list()

    hd_preds = {p.split('_')[0] for p in set(hd.predicates) - {hd.null_interaction}}
    hd_nodes = set(['hair_dryer' if obj == 'hair_drier' else obj for obj in hd.objects]) | hd_preds
    print(hd_nodes - set(cnet.nodes))

    # cnet.find_relations(hd_nodes, walk_length=1)
    cnet.find_relations(list(hd_nodes)[:3], walk_length=1)

    # cnet.export_to_rotate_edge_list('../RotatE/data/ConceptNet')


if __name__ == '__main__':
    main()
