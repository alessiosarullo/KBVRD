import json
import os
import pickle
from collections import Counter
from typing import Dict, Tuple

import numpy as np
from nltk.corpus import wordnet as wn

from config import cfg
from lib.dataset.hoi_dataset import HoiDataset


class ImSitu:
    def __init__(self):
        self.cache_file = os.path.join(cfg.cache_root, 'imsitu_dobjs.pkl')

        self.imsitu = ImSituDriver()

        # FIXME direct objects only? Doesn't work for example in "jump over an obstacle", though
        dobj_tokens_in_verb_abstracts = self.get_dobj_tokens_in_verb_abstracts()
        # for i, (verb, dobjs) in enumerate(dobj_tokens_in_verb_abstracts.items()):
        #     print('%3d %-15s > %-120s %s' % (i, verb, self.imsitu.verbs[verb]['abstract'], dobjs))

        self.abstract_dobj_per_verb = self.get_abstract_dobj_per_verb(dobj_tokens_in_verb_abstracts, sorted(self.imsitu.verbs.keys()))
        self.concrete_dobjs_count_per_verb = self.get_dobj_instances_per_verb(self.abstract_dobj_per_verb)

    def extract_freq_matrix(self, dataset: HoiDataset, return_known_mask=False):
        pred_verb_matches = self.match_preds_with_verbs(dataset)
        # print()
        # print('Matched: %d actions out of %d.' % (len(pred_verb_matches), len(dataset.actions)))
        # print('%-20s %10s  %s' % ('PREDICATE', 'VERB', 'ABSTRACT'))
        # for pred in dataset.actions:
        #     verb = pred_verb_matches.get(pred, None)
        #     v_str = ('%10s: %s' % (verb, self.imsitu.verbs[verb]['abstract'])) if verb is not None else ''
        #     print('%-20s %s' % (pred, v_str))
        #
        # print()
        # for i, pred in enumerate(dataset.actions):
        #     try:
        #         verb = pred_verb_matches[pred]
        #         abstract = self.imsitu.verbs[verb]['abstract']
        #         abstract_obj = self.abstract_dobj_per_verb[verb]
        #         abstract = abstract.split(abstract_obj)[0] + (abstract_obj if abstract_obj in abstract else '')
        #         v_str = '%-15s > %-15s %-50s' % (verb, abstract_obj, abstract)
        #     except KeyError:
        #         v_str = ''
        #     print('%3d %-20s %s' % (i, pred, v_str))
        #
        # print()
        # for i, pred in enumerate(dataset.actions):
        #     try:
        #         verb = pred_verb_matches[pred]
        #         instance_list = sorted(self.concrete_dobjs_count_per_verb[verb].keys())
        #         dobj = self.abstract_dobj_per_verb[verb]
        #     except KeyError:
        #         verb, instance_list, dobj = '', '', ''
        #     print('%3d %-20s %-15s %-15s > %-s' % (i, pred, verb, dobj, instance_list))

        matching_dobjs_count_per_verb = self.match_objects(dataset, self.concrete_dobjs_count_per_verb)
        # print()
        # for i, pred in enumerate(dataset.actions):
        #     try:
        #         verb = pred_verb_matches[pred]
        #         instance_list = matching_dobjs_count_per_verb[verb]
        #         dobj = abstract_dobj_per_verb[verb]
        #     except KeyError:
        #         verb, instance_list, dobj = '', '', ''
        #     print('%3d %-20s %-15s %-15s > %-s' % (i, pred, verb, dobj, instance_list if instance_list or not verb else '##########'))

        matching_dobjs_count_per_pred = {pred: matching_dobjs_count_per_verb.get(pred_verb_matches.get(pred), {}) for pred in dataset.actions}

        # Object-predicate matrix
        op_mat = np.array([[matching_dobjs_count_per_pred.get(pred, {}).get(obj, 0) for obj in dataset.objects]
                           for pred in dataset.actions], dtype=np.float).T
        if return_known_mask:
            imsitu_nouns = {g for noun in self.imsitu.nouns.values() for g in noun['gloss']}
            known_objects = np.array([obj in imsitu_nouns for obj in dataset.objects], dtype=bool)
            known_predicates = np.array([pred in pred_verb_matches.keys() for pred in dataset.actions], dtype=bool)
            known_mask = known_objects[:, None] & known_predicates[None, :]
            assert known_mask.shape == op_mat.shape
            assert np.all(known_mask[op_mat > 0])
            return op_mat, known_objects, known_predicates
        else:
            return op_mat

    def get_dobj_tokens_in_verb_abstracts(self):
        try:
            with open(self.cache_file, 'rb') as f:
                direct_objects_per_verb = pickle.load(f)
        except FileNotFoundError:
            import requests
            from nltk.parse.corenlp import CoreNLPServer, CoreNLPServerError, CoreNLPDependencyParser
            os.environ['CORENLP_HOME'] = os.path.abspath('CoreNLP')
            os.environ['CLASSPATH'] = os.path.abspath('CoreNLP')
            os.environ['JAVAHOME'] = 'C:/Program Files/Java/jdk-10.0.2/bin/java.exe'  # FIXME

            verbs = self.imsitu.verbs

            direct_objects_per_verb = {}
            while set(direct_objects_per_verb.keys()) != set(verbs.keys()):
                verbs_to_parse = set(verbs.keys()) - set(direct_objects_per_verb.keys())
                server = CoreNLPServer()
                server_started = False
                try:
                    try:
                        server.start()
                        print('CoreNLP server started.')
                        server_started = True
                    except CoreNLPServerError:
                        print("Couldn't start CoreNLP server. Possible reason: already running.")

                    # parser = CoreNLPParser(url='http://localhost:9000')  # Constituency parsing
                    parser = CoreNLPDependencyParser(url='http://localhost:9000')  # Dependency parsing
                    for verb in verbs_to_parse:
                        abstract = verbs[verb]['abstract']
                        abstract = ' '.join([token.split('/')[0] for token in abstract.split()])  # 'an AGENT jumps over/through an OBSTACLE'
                        assert abstract.count('(') == abstract.count(')') <= 1
                        if '(' in abstract:  # 'the SLIDER (when different from the AGENT) on'
                            abstract = abstract.split('(')[0].strip() + ' ' + abstract.split(')')[1].strip()
                        parse_result = list(next(parser.raw_parse(abstract)).triples())
                        direct_objects_per_verb[verb] = [(src[0], dst[0]) for src, dep, dst in parse_result if dep == 'dobj']
                except requests.exceptions.ReadTimeout:
                    pass
                finally:
                    if server_started:
                        server.stop()
                        print('CoreNLP server stopped.')
                print('Parsed: %d/%d' % (len(direct_objects_per_verb), len(verbs)))

            with open(self.cache_file, 'wb') as f:
                pickle.dump(direct_objects_per_verb, f)
        return direct_objects_per_verb

    def get_abstract_dobj_per_verb(self, dobj_tokens_in_verb_abstracts, verbs):
        abstract_dobj_per_verb = {}
        for verb in verbs:
            abstract = self.imsitu.verbs[verb]['abstract']
            dobjs = dobj_tokens_in_verb_abstracts[verb]
            if dobjs:
                abstract_verb, abstract_obj = dobjs[0]
                if abstract_obj == 'agentparts':  # FIXME
                    abstract_obj = abstract_obj[:-1]
                abstract_obj = abstract_obj.upper() if abstract_obj.upper() in abstract else abstract_obj
                abstract_dobj_per_verb[verb] = abstract_obj
        return abstract_dobj_per_verb

    def get_dobj_instances_per_verb(self, abstract_dobj_per_verb, split='train'):
        anns = self.imsitu.get_annotations(split)
        concrete_dobjs_per_verb = {}
        for ann in anns.values():
            verb = ann['verb']
            verb_dobj = abstract_dobj_per_verb.get(verb, None)
            if verb_dobj:
                for frame in ann['frames']:
                    instance = frame.get(verb_dobj.lower(), None)
                    if instance:  # might be empty or None
                        concrete_dobjs_per_verb.setdefault(verb, []).append(instance)
        concrete_dobjs_per_verb = {verb: Counter([instance for iid in instance_ids for instance in self.imsitu.nouns[iid]['gloss']])
                                   for verb, instance_ids in concrete_dobjs_per_verb.items()}
        return concrete_dobjs_per_verb

    def match_preds_with_verbs(self, dataset: HoiDataset):
        verb_dict = self.imsitu.verbs
        pred_verb_matches = {}
        for orig_pred in dataset.actions:
            pred = orig_pred
            if pred != dataset.null_action:
                pred = pred.replace('_', ' ')
            pred_verb = pred.split()[0]

            matches_in_key = list(set([verb for verb in verb_dict.keys() if pred_verb == wn.morphy(verb, wn.VERB)]))
            match = None
            if not matches_in_key:
                matches_in_abstract = [verb for verb, ventry in verb_dict.items() if (' ' + pred_verb) in ventry['abstract']]
                if matches_in_abstract and len(matches_in_abstract) == 1:
                    match = matches_in_abstract[0]
            else:
                assert len(matches_in_key) == 1
                match = matches_in_key[0]

            if match is not None:
                pred_verb_matches[orig_pred] = match
        return pred_verb_matches

    @staticmethod
    def match_objects(dataset, concrete_dobjs_per_verb: Dict[str, Counter]):
        # TODO instead of only matching objects in hico-det some more refined procedure could be applied. For example, matching considering
        # embedding similarity or definitions
        hico_obj_set = set(dataset.objects)
        matching_dobjs_per_verb = {verb: {instance: count for instance, count in instances.items() if instance in hico_obj_set}
                                   for verb, instances in concrete_dobjs_per_verb.items()}
        return matching_dobjs_per_verb


class ImSituDriver:
    def __init__(self):
        """
        Attributes:
            - nouns [dict(dict)]: More than 80k entries. Keys are ImageNet synsets, which are in turn derived from WordNet 3.0. Values are
                dictionaries containing the following items:
                    - 'gloss' [list(str)]: List of nouns  describing the concept.
                    - 'def' [str]: A definition.
                EXAMPLE: Key: 'n03024882'. Value:
                    {'gloss': ['choker', 'collar', 'dog collar', 'neckband'],
                     'def': "necklace that fits tightly around a woman's neck"
                     }
            - verbs [dict]: Around 500 entries. Keys are verbs themselves [str], while values are dictionaries of:
                - 'framenet' [str]: ID of the verb in FrameNet. Seems to somehow describe the category the verb belongs to.
                - 'def' [str]: Definition of the verb.
                - 'roles' [dict]: A dictionaries of the roles involved in the action specified by this verb. Keys vary according to the verb and
                    each item contains:
                        - 'framenet': See above.
                        - 'def' [str]: Describes the role the item specified by this key has.
                - 'abstract' [str]: A string describing the action on a general level.
                - 'order' [list(str)]: The order the roles appear in `abstract`.
                EXAMPLE: Key: 'tattoing'. Value:
                    {'framenet': 'Create_physical_artwork',
                     'def': 'to mark the skin with permanent colors and patterns',
                     'roles':
                        {'tool': {'framenet': 'instrument', 'def': 'The tool used'},
                         'place': {'framenet': 'place', 'def': 'The location where the tattoo event is happening'},
                         'target': {'framenet': 'representation', 'def': 'The entity being tattooed'},
                         'agent': {'framenet': 'creator', 'def': 'The entity doing the tattoo action'}
                         }
                     'abstract': 'AGENT tattooed TARGET with TOOL in PLACE',
                     'order': ['agent', 'target', 'tool', 'place'],
                    }
            - train, val, test [dict(dict)]: Keys are image file names, values are dictionaries with the following keys:
                - 'verb' [str]: Verb describing the image. It's a key for `verbs`.
                - 'frames' [list(dict)]: Each item is a dictionary. Keys are the roles specified in `verbs` for this verb, taking their values from
                    `nouns`'s keys.
                EXAMPLE: Key: 'glaring_215.jpg'. Value:
                    {'verb': 'glaring',
                     'frames': [{'place': 'n04215402', 'perceiver': '', 'agent': 'n10287213'},
                                {'place': 'n08613733', 'perceiver': '', 'agent': 'n10287213'},
                                {'place': 'n08613733', 'perceiver': '', 'agent': 'n10287213'}
                                ]
                    }
        """
        data_dir = os.path.join(cfg.data_root, 'imSitu')
        self.image_dir = os.path.join(data_dir, 'images')
        self.path_domain_file = os.path.join(data_dir, 'imsitu_space.json')
        self.path_train_file = os.path.join(data_dir, 'train.json')
        self.path_val_file = os.path.join(data_dir, 'dev.json')
        self.path_test_file = os.path.join(data_dir, 'test.json')

        self.nouns, self.verbs, self.train, self.val, self.test = self.load()
        self.verbs = self.fix_verbs(self.verbs)

    @staticmethod
    def fix_verbs(verbs) -> Dict[str, Dict]:
        verbs['riding']['abstract'] = verbs['riding']['abstract'].replace(' then ', ' the ')
        verbs['teaching']['abstract'] = verbs['teaching']['abstract'].replace(' to teach ', ' teaches ')
        return verbs

    def get_annotations(self, split):
        splits = {'train': self.train,
                  'val': self.val,
                  'test': self.test,
                  }
        return splits[split]

    def _print_verb_entry(self, verb):
        print('ImSitu verb entry example:')
        for k, v in self.verbs[verb].items():
            if k != 'roles':
                print('%15s: %s' % (k, v))
            else:
                print('%15s:' % k)
                ln = max([len(r) for r in v.keys()]) + 1
                for r, d in v.items():
                    print(('%15s  - %-' + str(ln) + 's %s') % ('', r + ':', d))

    def load(self) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict], Dict[str, Dict], Dict[str, Dict]]:
        with open(self.path_domain_file, 'r') as f:
            domain = json.load(f)
        verbs, nouns = domain['verbs'], domain['nouns']

        with open(self.path_train_file, 'r') as f:
            train = json.load(f)
        with open(self.path_val_file, 'r') as f:
            val = json.load(f)
        with open(self.path_test_file, 'r') as f:
            test = json.load(f)

        return nouns, verbs, train, val, test


def main():
    from lib.dataset.hico import Hico
    imsitu = ImSituDriver()
    imsitu._print_verb_entry('riding')

    # imsitu_ke = ImSitu()
    # hico = Hico()
    # imsitu_op_mat, known_objects, known_predicates = imsitu_ke.extract_freq_matrix(hico, return_known_mask=True)


if __name__ == '__main__':
    main()
