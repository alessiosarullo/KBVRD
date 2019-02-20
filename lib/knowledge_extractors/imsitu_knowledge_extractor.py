import os
import pickle

import numpy as np
import requests
from drivers.imsitu_driver import ImSitu
from nltk.parse.corenlp import CoreNLPServer, CoreNLPServerError, CoreNLPDependencyParser
from analysis.utils import plot_mat

from lib.dataset.hicodet_driver import HicoDet


# TODO check after Hico predicate refactor
class ImSituKnowledgeExtractor:
    def __init__(self):
        self.hd = HicoDet()
        self.imsitu = ImSitu()

        # FIXME direct objects only? Doesn't work for example in "jump over an obstacle"
        dobj_tokens_in_verb_abstracts = self.get_dobj_tokens_in_verb_abstracts()
        # for i, (verb, dobjs) in enumerate(dobj_tokens_in_verb_abstracts.items()):
        #     print('%3d %-15s > %-120s %s' % (i, verb, self.imsitu.verbs[verb]['abstract'], dobjs))

        pred_verb_matches = self.match_preds_with_verbs()
        # print('%-20s   %10s  %s' % ('PREDICATE', 'VERB', 'ABSTRACT'))
        # for pred_id, verb in pred_verb_matches.items():
        #     print('%-20s | %10s: %s' % (self.hd.predicate_dict[pred_id]['name'], verb, self.imsitu.verbs[verb]['abstract']))

        abstract_dobj_per_verb = self.get_abstract_dobj_per_matched_verb(dobj_tokens_in_verb_abstracts, pred_verb_matches)
        # for i, (pred_idx, pred) in enumerate(self.hd.predicate_dict.items()):
        #     verb = ''
        #     abstract = ''
        #     abstract_verb, abstract_obj = '', ''
        #     try:
        #         verb = pred_verb_matches[pred_idx]
        #         abstract = self.imsitu.verbs[verb]['abstract']
        #         abstract_obj = abstract_dobj_per_verb[verb]
        #         abstract = abstract.split(abstract_obj)[0] + (abstract_obj if abstract_obj in abstract else '')
        #     except KeyError:
        #         pass
        #     print('%3d %-15s %-15s > %-50s %-s' % (i, pred['name'], verb, abstract, abstract_obj))

        print()
        concrete_dobjs_per_verb = self.get_dobj_instances(abstract_dobj_per_verb)
        # for i, (pred_idx, pred) in enumerate(self.hd.predicate_dict.items()):
        #     try:
        #         verb = pred_verb_matches[pred_idx]
        #         instance_list = concrete_dobjs_per_verb[verb]
        #         dobj = abstract_dobj_per_verb[verb]
        #     except KeyError:
        #         verb, instance_list, dobj = '', '', ''
        #     print('%3d %-15s %-15s %-15s > %-s' % (i, pred['name'], verb, dobj, instance_list))

        print()
        matching_dobjs_per_verb = self.match_objects(concrete_dobjs_per_verb)
        for i, (pred, pred_entry) in enumerate(self.hd.predicate_dict.items()):
            try:
                verb = pred_verb_matches[pred]
                instance_list = matching_dobjs_per_verb[verb]
                dobj = abstract_dobj_per_verb[verb]
            except KeyError:
                verb, instance_list, dobj = '', '', ''
            print('%3d %-15s %-15s %-15s > %-s' % (i, pred_entry['name'], verb, dobj, instance_list if instance_list or not verb else '##########'))

        matching_dobjs_per_pred = {pred: matching_dobjs_per_verb.get(pred_verb_matches.get(pred), []) for pred in self.hd.predicate_dict}

        # imSitu object-predicate matrix
        imsitu_op_mat = np.array([[1 if obj in matching_dobjs_per_pred.get(pred, []) else 0 for obj in self.hd.objects]
                                  for pred in self.hd.predicate_dict.keys()]).T

        # Hico object-predicate matrix
        # TODO move somewhere else and call it, as it's used in a few different places
        pred_to_idx = {k: i for i, k in enumerate(self.hd.predicate_dict.keys())}
        obj_to_idx = {o: i for i, o in enumerate(self.hd.objects)}
        hico_op_mat = np.zeros([len(obj_to_idx), len(pred_to_idx)])
        for inter in self.hd.interactions:
            hico_op_mat[obj_to_idx[inter['obj']], pred_to_idx[inter['pred']]] = 1

        # Plot
        plot_mat((imsitu_op_mat + hico_op_mat * 2) / 3, self.hd.predicates, self.hd.objects)

    def match_objects(self, concrete_dobjs_per_verb):
        # TODO instead of only matching objects in hico-det some more refined procedure could be applied. For example, matching considering
        # embedding similarity or definitions
        hd_obj_set = set(self.hd.objects)
        matching_dobjs_per_verb = {verb: [instance for instance in instances if instance in hd_obj_set]
                                   for verb, instances in concrete_dobjs_per_verb.items()}
        return matching_dobjs_per_verb

    def get_dobj_instances(self, abstract_dobj_per_verb, split='train'):
        anns = self.imsitu.get_annotations(split)
        concrete_dobjs_per_verb = {}
        for ann in anns.values():
            verb = ann['verb']
            verb_dobj = abstract_dobj_per_verb.get(verb, None)
            if verb_dobj:
                for frame in ann['frames']:
                    instance = frame[verb_dobj.lower()]
                    if instance:  # might be empty
                        concrete_dobjs_per_verb.setdefault(verb, set()).add(instance)
        concrete_dobjs_per_verb = {verb: sorted(set([instance for iid in instance_ids for instance in self.imsitu.nouns[iid]['gloss']]))
                                   for verb, instance_ids in concrete_dobjs_per_verb.items()}
        return concrete_dobjs_per_verb

    def get_abstract_dobj_per_matched_verb(self, dobj_tokens_in_verb_abstracts, pred_verb_matches):
        abstract_dobj_per_verb = {}
        for verb in pred_verb_matches.values():
            abstract = self.imsitu.verbs[verb]['abstract']
            dobjs = dobj_tokens_in_verb_abstracts[verb]
            if dobjs:
                abstract_verb, abstract_obj = dobjs[0]
                if abstract_obj == 'agentparts':  # FIXME
                    abstract_obj = abstract_obj[:-1]
                abstract_obj = abstract_obj.upper() if abstract_obj.upper() in abstract else abstract_obj
                abstract_dobj_per_verb[verb] = abstract_obj
        return abstract_dobj_per_verb

    def match_preds_with_verbs(self):
        pred_dict = self.hd.predicate_dict
        verb_dict = self.imsitu.verbs

        pred_verb_matches = {}
        for pred, pred_entry in pred_dict.items():
            ing = pred_entry['ing']
            if pred != 'no_interaction':
                pred = pred.replace('_', ' ')
                ing = ing.replace('_', ' ').split()[0]
            matches_in_key = list(set([verb for verb in verb_dict.keys() if ing == verb]))
            if not matches_in_key:
                matches_in_abstract = [verb for verb, ventry in verb_dict.items() if ' ' + pred.split()[0] in ventry['abstract']]
                if matches_in_abstract:
                    assert len(matches_in_abstract) == 1
                    pred_verb_matches[pred] = matches_in_abstract[0]
            else:
                assert len(matches_in_key) == 1
                pred_verb_matches[pred] = matches_in_key[0]
        print('Matched: %d predicates out of %d.' % (len(pred_verb_matches), len(pred_dict)))

        return pred_verb_matches

    def get_dobj_tokens_in_verb_abstracts(self):
        cache_file = os.path.join('cache', 'dobjs.pkl')
        try:
            with open(cache_file, 'rb') as f:
                direct_objects_per_verb = pickle.load(f)
        except FileNotFoundError:
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

            with open(cache_file, 'wb') as f:
                pickle.dump(direct_objects_per_verb, f)
        return direct_objects_per_verb


def main():
    ImSituKnowledgeExtractor()


if __name__ == '__main__':
    main()
