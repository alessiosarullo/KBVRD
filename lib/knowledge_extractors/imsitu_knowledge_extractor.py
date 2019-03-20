import os
import pickle
from collections import Counter
from typing import Dict

import numpy as np

from config import cfg
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.dataset.imsitu_driver import ImSitu


class ImSituKnowledgeExtractor:
    def __init__(self):
        self.cache_file = os.path.join(cfg.program.cache_root, 'imsitu_dobjs.pkl')

        self.imsitu = ImSitu()

        # FIXME direct objects only? Doesn't work for example in "jump over an obstacle", though
        dobj_tokens_in_verb_abstracts = self.get_dobj_tokens_in_verb_abstracts()
        # for i, (verb, dobjs) in enumerate(dobj_tokens_in_verb_abstracts.items()):
        #     print('%3d %-15s > %-120s %s' % (i, verb, self.imsitu.verbs[verb]['abstract'], dobjs))

        self.abstract_dobj_per_verb = self.get_abstract_dobj_per_verb(dobj_tokens_in_verb_abstracts, sorted(self.imsitu.verbs.keys()))
        self.concrete_dobjs_count_per_verb = self.get_dobj_instances_per_verb(self.abstract_dobj_per_verb)

    def extract_prior_matrix(self, dataset: HicoDetInstanceSplit):
        pred_verb_matches = self.match_preds_with_verbs(dataset)
        # print()
        # print('Matched: %d predicates out of %d.' % (len(pred_verb_matches), len(dataset.predicates)))
        # print('%-20s %10s  %s' % ('PREDICATE', 'VERB', 'ABSTRACT'))
        # for pred in dataset.predicates:
        #     verb = pred_verb_matches.get(pred, None)
        #     v_str = ('%10s: %s' % (verb, self.imsitu.verbs[verb]['abstract'])) if verb is not None else ''
        #     print('%-20s %s' % (pred, v_str))
        #
        # print()
        # for i, pred in enumerate(dataset.predicates):
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
        # for i, pred in enumerate(dataset.predicates):
        #     try:
        #         verb = pred_verb_matches[pred]
        #         instance_list = self.concrete_dobjs_per_verb[verb]
        #         dobj = self.abstract_dobj_per_verb[verb]
        #     except KeyError:
        #         verb, instance_list, dobj = '', '', ''
        #     print('%3d %-20s %-15s %-15s > %-s' % (i, pred, verb, dobj, instance_list))

        matching_dobjs_count_per_verb = self.match_objects(dataset, self.concrete_dobjs_count_per_verb)
        # print()
        # for i, pred in enumerate(dataset.predicates):
        #     try:
        #         verb = pred_verb_matches[pred]
        #         instance_list = matching_dobjs_count_per_verb[verb]
        #         dobj = abstract_dobj_per_verb[verb]
        #     except KeyError:
        #         verb, instance_list, dobj = '', '', ''
        #     print('%3d %-20s %-15s %-15s > %-s' % (i, pred, verb, dobj, instance_list if instance_list or not verb else '##########'))

        matching_dobjs_count_per_pred = {pred: matching_dobjs_count_per_verb.get(pred_verb_matches.get(pred), {}) for pred in dataset.predicates}

        # Object-predicate matrix
        op_mat = np.array([[matching_dobjs_count_per_pred.get(pred, {}).get(obj, 0) for obj in dataset.objects]
                           for pred in dataset.predicates], dtype=np.float).T
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
            verbs = {k: v for k, v in verbs.items() if k in ['hitting', 'flipping', 'jumping', 'sliding']}

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

    def match_preds_with_verbs(self, dataset: HicoDetInstanceSplit):
        verb_dict = self.imsitu.verbs
        pred_verb_matches = {}
        for orig_pred in dataset.predicates:
            ing = dataset.hicodet.predicate_dict[orig_pred]['ing']
            pred = orig_pred
            if pred != dataset.hicodet.null_interaction:
                pred = pred.replace('_', ' ')
                ing = ing.replace('_', ' ').split()[0]

            matches_in_key = list(set([verb for verb in verb_dict.keys() if ing == verb]))
            match = None
            if not matches_in_key:
                matches_in_abstract = [verb for verb, ventry in verb_dict.items() if ' ' + pred.split()[0] in ventry['abstract']]
                if matches_in_abstract:
                    assert len(matches_in_abstract) == 1
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
        hd_obj_set = set(dataset.objects)
        matching_dobjs_per_verb = {verb: {instance: count for instance, count in instances.items() if instance in hd_obj_set}
                                   for verb, instances in concrete_dobjs_per_verb.items()}
        return matching_dobjs_per_verb
