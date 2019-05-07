import json
import os
import pickle
import scipy.sparse
import numpy as np
import re

from config import cfg


class Conceptnet:
    def __init__(self, edges=None):
        self.pos_tags = ['n', 'v', 'a', 's', 'r']
        self.rels_to_filter = ['Antonym', 'DistinctFrom', 'NotCapableOf', 'NotDesires', 'NotHasProperty',
                               'dbpedia/capital', 'dbpedia/field', 'dbpedia/genre', 'dbpedia/genus', 'dbpedia/influencedBy',
                               'dbpedia/knownFor', 'dbpedia/language', 'dbpedia/leader', 'dbpedia/occupation', 'dbpedia/product']

        data_dir = os.path.join(cfg.program.data_root, 'ConceptNet')
        self.path_cnet = os.path.join(cfg.program.cache_root, 'conceptnet.pkl')
        self.path_raw_cnet = os.path.join(data_dir, 'raw_conceptnet.pkl')
        self.path_raw_cnet_eng = os.path.join(data_dir, 'conceptnet560_en.txt')
        self.path_raw_cnet_orig = os.path.join(data_dir, 'conceptnet560.csv')

        if edges is None:
            self.edges = self._load()
        else:
            self.edges = edges
        self.nodes = self.edges_from = self.edges_to = self.rel_occurrs = self.relations = None
        self._init()

    def has_pos_tag(self, entry, tag=None):
        if len(entry) > 2 and entry[-2] == '/':
            assert entry[-1] in self.pos_tags
            if tag is None:
                return True
            else:
                return entry[-1] == tag
        return False

    def get_adjacency_matrix(self):
        num_nodes = len(self.nodes)
        if num_nodes**2 >= 1e10:
            print('Number of nodes is too big: %d.' % num_nodes)
            return
        node_index = {n: i for i, n in enumerate(self.nodes)}
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float16)
        for src in self.nodes:
            i = node_index[src]
            for e_idx in self.edges_from.get(src, []):
                j = node_index[self.edges[e_idx]['dst']]
                adj[i, j] = 1
        adj[np.arange(num_nodes), np.arange(num_nodes)] = 0
        return adj

    # Iterator methods
    def __iter__(self):
        return self.edges

    def __next__(self):
        return self.edges.__next__()

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, item):
        return self.edges[item]

    # Filter
    def filter_nodes(self, node_seed, radius=3):
        node_seed = set(['%s/%s' % (n, tag) for n in node_seed for tag in self.pos_tags]) | set(node_seed)
        node_set = set(self.nodes)
        keep = node_seed & node_set
        neighs_r = keep
        for r in range(1, radius):
            neighs_r = {self.edges[e]['dst'] for n in neighs_r for e in self.edges_from.get(n, [])} - keep
            keep = keep | neighs_r

        self.edges = [e for e in self.edges if e['src'] in keep and e['dst'] in keep]
        self._init()

    def find_relations(self, src_nodes, walk_length=1):
        node_variants_index = {(n + tag): i for i, n in enumerate(src_nodes) for tag in [''] + ['/' + t for t in self.pos_tags]}
        cnet_nodes_index = {n: i for i, n in enumerate(self.nodes)}

        nodes_of_interest = list(set(node_variants_index.keys()) & set(self.nodes))
        inds = np.array([cnet_nodes_index[h] for h in nodes_of_interest])

        adj = self.get_adjacency_matrix()
        rel = np.zeros((inds.size, inds.size))
        left = adj[inds, :]  # this is equivalent to diag(inds) @ adj, but more computationally efficient
        right = adj[:, inds]
        if walk_length >= 1:
            rel += adj[inds, :][:, inds]  # similarly, equivalent to diag(inds) @ adj @ diag(inds)
        for step in range(2, walk_length):
            # Equivalent to diag(inds) @ adj^step @ diag(inds)
            rel += left @ right

            # The minimum is there because we don't care about how many paths are there
            if step + 1 < walk_length:
                if step % 2 == 0:
                    left = np.minimum(1, left @ adj)
                else:
                    right = np.minimum(1, adj @ right)
        rel[np.arange(inds.size), np.arange(inds.size)] = 0
        rel = np.minimum(1, rel)

        src_to_inds = np.zeros((len(src_nodes), inds.size))
        for i, idx in enumerate(inds):
            src_to_inds[node_variants_index[self.nodes[idx]], i] = 1
        rel = np.minimum(1, src_to_inds @ rel @ src_to_inds.T)
        return rel

    # Export methods
    def export_to_deepwalk_edge_list(self, relationship=None):
        node_index = {n: i for i, n in enumerate(self.nodes)}
        output = []
        for edge in self.edges:
            if relationship is None or edge['rel'] == relationship:
                output.append('%d %d' % (node_index[edge['src']], node_index[edge['dst']]))
        lines = '\n'.join(output)
        with open(os.path.join(cfg.program.cache_root, 'cnet.edgelist'), 'w') as f:
            f.write(lines)

    def export_to_rotate_edge_list(self, output_dir, relationship=None):
        output = []
        for edge in self.edges:
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

    def _init(self):
        self.nodes = sorted({e[k] for e in self.edges for k in ['src', 'dst']})

        self.edges_from = {}
        self.edges_to = {}
        rel_occurrs = {}
        for i, e in enumerate(self.edges):
            self.edges_from.setdefault(e['src'], []).append(i)
            self.edges_to.setdefault(e['dst'], []).append(i)

            rel = e['rel']
            rel_occurrs[rel] = rel_occurrs.get(rel, 0) + 1

        self.rel_occurrs = {r: rel_occurrs[r] for r in sorted(rel_occurrs.keys(), key=lambda x: rel_occurrs[x], reverse=True)}
        self.relations = sorted(set([edge['rel'] for edge in self.edges]))

    def _extend_and_filter(self, edges, mandatory_pos_tag=False):
        def _hash(_e):
            return '|||'.join([str(_v) for _v in _e.values()])

        edges_hash = {_hash(edge) for edge in edges}
        new_edges = []
        to_keep = []
        for i, e in enumerate(edges):
            src, rel, dst = e['src'], e['rel'], e['dst']
            # Filter
            if rel in self.rels_to_filter or \
                    (mandatory_pos_tag and not (self.has_pos_tag(src) and self.has_pos_tag(dst))) or \
                    any([bool(re.search(r"[^a-zA-Z0-9_'\-]", (word[:-2] if self.has_pos_tag(word) else word))) for word in (src, dst)]):
                continue

            # Extend
            to_keep.append(i)
            if rel == 'RelatedTo':
                new_edge = {'src': dst,
                            'rel': 'RelatedTo',
                            'dst': src,
                            'weight': e['weight'],
                            }
                if _hash(new_edge) not in edges_hash:
                    new_edges.append(new_edge)

            if rel == 'IsA':
                new_edges.append({'src': dst,
                                  'rel': 'hasSubtype',
                                  'dst': src,
                                  'weight': e['weight'],
                                  })
        return [edges[i] for i in to_keep] + new_edges

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


def main():
    from lib.dataset.hicodet_driver import HicoDet
    hd = HicoDet()

    with open('cache/cnet_hd2.pkl', 'rb') as f:
        cnet_edges = pickle.load(f)
    cnet = Conceptnet(edges=cnet_edges)
    print(cnet.nodes[:5])

    # cnet.export_to_deepwalk_edge_list()

    hd_preds = {noun.split('_')[0] for noun in set(hd.predicates) - {hd.null_interaction}}
    hd_nodes = set(['hair_dryer' if obj == 'hair_drier' else obj for obj in hd.objects]) | hd_preds
    print(hd_nodes - set(cnet.nodes))

    # cnet.find_relations(hd_nodes, walk_length=1)
    cnet.find_relations(list(hd_nodes)[:3], walk_length=1)

    # cnet.export_to_rotate_edge_list('../RotatE/data/ConceptNet')

    return cnet, hd, hd_nodes


if __name__ == '__main__':
    main()
