import json
import os
import pickle
import numpy as np


class Conceptnet:
    def __init__(self):
        self.POS_TAGS = ['n', 'v', 'a', 's', 'r']
        self.PATH_SEP = '#'
        self.RELS_TO_FILTER = ['Antonym', 'DistinctFrom', 'NotCapableOf', 'NotDesires', 'NotHasProperty',
                               'dbpedia/capital', 'dbpedia/field', 'dbpedia/genre', 'dbpedia/genus', 'dbpedia/influencedBy',
                               'dbpedia/knownFor', 'dbpedia/language', 'dbpedia/leader', 'dbpedia/occupation', 'dbpedia/product']
        self.USELESS_RELS = ['MannerOf', 'EtymologicallyRelatedTo', 'DerivedFrom', 'AtLocation',
                             'PartOf', 'IsA']

        data_dir = os.path.join('data', 'ConceptNet')
        self.path_cnet = os.path.join(data_dir, 'conceptnet.pkl')
        self.path_raw_cnet = os.path.join(data_dir, 'raw_conceptnet.pkl')
        self.path_raw_cnet_eng = os.path.join(data_dir, 'conceptnet560_en.txt')
        self.path_raw_cnet_orig = os.path.join(data_dir, 'conceptnet560.csv')

        self.edges = self.load()
        self.nodes, self.edges_from, self.edges_to, self.rel_occurrs = self.build_cache()

        self._relations = None
        self._filtered_rels = None

    def find_outgoing_edges_inds(self, query, include_unk_pos=True):
        if not self.has_pos_tag(query):
            queries = ['%s/%s' % (query, pt) for pt in self.POS_TAGS]
            if include_unk_pos:
                queries += [query]
        else:
            queries = [query]

        results = []
        for q in queries:
            results.extend([i for i in self.edges_from.get(q, [])])
        return results

    def find_pos_forms(self, query):
        forms = set()
        for e in self.edges:
            for field in ['src', 'dst']:
                if e[field] == query or e[field].split('/')[0] == query:
                    forms.add(e[field])
        return list(forms)

    def find_paths(self, src_node, dst_nodes, max_length=3, best_path_only=True):
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
                edge_inds = self.find_outgoing_edges_inds(node)
                for edge_idx in edge_inds:
                    next_node = self.edges[edge_idx]['dst']
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
                    path_str = self.edges[path[0]]['src']
                    for edge_idx in path:
                        edge = self.edges[edge_idx]
                        path_str += '%s%s (%4.2f)%s%s' % (self.PATH_SEP, edge['rel'], edge['weight'], self.PATH_SEP, edge['dst'])
                    solution_paths.setdefault(obj, set()).add(path_str)

        # Filter paths
        solution_paths = {obj: [path for path in paths if all(bad_rel not in path for bad_rel in self.USELESS_RELS)]
                          for obj, paths in solution_paths.items()}
        solution_paths = {k: v for k, v in solution_paths.items() if v}

        if best_path_only:
            best_paths = {}
            for obj, paths in solution_paths.items():
                scores = [sum([float(rel_token[:-1].split('(')[-1]) * 1 / (p + 1)
                               for p, rel_token in enumerate(path.split(self.PATH_SEP)[1::2])])
                          for path in paths]
                best_paths[obj] = [paths[np.argmax(scores)]]
            return best_paths
        else:
            return {k: sorted(v) for k, v in solution_paths.items()}

    def load(self):
        try:
            with open(self.path_cnet, 'rb') as f:
                print('Loading the refined ConceptNet')
                edges = pickle.load(f)
                print('Done')
        except FileNotFoundError:
            edges = self.load_raw()
            print('Extending')
            edges = self.extend_and_filter(edges)  # lowercase relationships are manually added
            with open(self.path_cnet, 'wb') as f:
                pickle.dump(edges, f)
            print('Done')
        return edges

    def build_cache(self):
        nodes = sorted({e[k] for e in self.edges for k in ['src', 'dst']})

        edges_from = {}
        edges_to = {}
        rel_occurrs = {}
        for i, e in enumerate(self.edges):
            edges_from.setdefault(e['src'], []).append(i)
            edges_to.setdefault(e['dst'], []).append(i)

            rel = e['rel']
            rel_occurrs[rel] = rel_occurrs.get(rel, 0) + 1

        rel_occurrs = {r: rel_occurrs[r] for r in sorted(rel_occurrs.keys(), key=lambda x: rel_occurrs[x], reverse=True)}
        return nodes, edges_from, edges_to, rel_occurrs

    def extend_and_filter(self, edges, mandatory_pos_tag=False):
        def _hash(_e):
            return '|||'.join([str(_v) for _v in _e.values()])

        edges_hash = {_hash(edge) for edge in edges}
        new_edges = []
        to_keep = []
        for i, e in enumerate(edges):
            src, rel, dst = e['src'], e['rel'], e['dst']
            if rel in self.RELS_TO_FILTER or (mandatory_pos_tag and not (self.has_pos_tag(src) and self.has_pos_tag(dst))):
                continue

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

    def load_raw(self):
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
                print('Done')
        except FileNotFoundError:
            print('Converting ConceptNet')
            # Only consider english entries
            try:
                with open(self.path_raw_cnet_eng, 'r', encoding='utf-8') as f:
                    entries = f.readlines()
            except FileNotFoundError:
                print('Filtering ENG entries . . .')
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
                print('Done.')

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
            print('Done.')
        return parsed_entries

    @property
    def relations(self):
        if self._relations is None:
            self._relations = sorted(set([edge['rel'] for edge in self.edges]))
        return self._relations

    @property
    def hico_rels(self):
        if self._filtered_rels is None:
            self._filtered_rels = [r for r in self.relations if r not in self.RELS_TO_FILTER]
        return self._filtered_rels

    def has_pos_tag(self, entry, tag=None):
        if len(entry) > 2 and entry[-2] == '/':
            assert entry[-1] in self.POS_TAGS
            if tag is None:
                return True
            else:
                return entry[-1] == tag
        return False

    # Iterator methods
    def __iter__(self):
        return self.edges

    def __next__(self):
        return self.edges.__next__()

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, item):
        return self.edges[item]


def main():
    Conceptnet()


if __name__ == '__main__':
    main()
