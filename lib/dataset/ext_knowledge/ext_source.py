import json
import os
import pickle
from typing import List

import numpy as np
from nltk.corpus import wordnet as wn

from config import cfg
from lib.dataset.hoi_dataset import HoiDataset


class ExtSource:
    def __init__(self):
        triplets_str = self._load()

        self.objects = sorted({t[i] for t in triplets_str for i in [0, 2]})
        self.object_index = {x: i for i, x in enumerate(self.objects)}
        self.predicates = sorted({t[1] for t in triplets_str})
        self.predicate_index = {x: i for i, x in enumerate(self.predicates)}

        self.triplets = np.array([[self.object_index[s], self.predicate_index[p], self.object_index[o]] for s, p, o in triplets_str])

    @property
    def human_classes(self) -> List[int]:
        raise NotImplementedError

    @property
    def triplet_str(self):
        return [(self.objects[s], self.predicates[p], self.objects[o]) for s, p, o in self.triplets]

    def _load(self):
        raise NotImplementedError

    def _parse_captions(self, captions, required_words=None):
        from allennlp.predictors.predictor import Predictor

        if required_words is not None:
            rws = set(required_words)
            filtered_captions = []
            for c in captions:
                cs = c.split()
                for rw in rws:
                    if rw in cs:
                        filtered_captions.append(c)
                        break
            captions = filtered_captions

        oie = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")
        cparser = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")

        triplets_str = []

        for i, c in enumerate(captions):
            if i % 100 == 0:
                print(f'{i:6d}/{len(captions)}')

            out = oie.predict(sentence=c)
            for verb in out['verbs']:
                d = verb['description']
                tagged_tokens = [[y.strip() for y in x.strip().split('[')[1].split(':')] for x in d.split(']') if x and '[' in x]
                tagged_tokens_dict = {tt[0]: tt[1] for tt in tagged_tokens}

                try:
                    s = tagged_tokens_dict['ARG0']
                    p = tagged_tokens_dict['V']
                    o = tagged_tokens_dict['ARG1']
                except KeyError:
                    continue

                cp_s = cparser.predict(sentence=s)
                s = ' '.join([tok for tok, tag in zip(cp_s['tokens'], cp_s['pos_tags']) if tag == 'NN'])

                cp_o = cparser.predict(sentence=o)
                o = ' '.join([tok for tok, tag in zip(cp_o['tokens'], cp_o['pos_tags']) if tag == 'NN'])

                try:
                    s = wn.morphy(s, wn.NOUN)
                    if s is not None:
                        if wn.morphy(p.split()[0], wn.VERB) == 'be':
                            p = ' '.join(p.split()[1:])
                        triplets_str.append([s, p, o])
                except:
                    continue

        return triplets_str

    #
    # def _parse_captions(self, captions, required_words=None):
    #     from stanfordnlp.server import CoreNLPClient
    #     os.environ['CORENLP_HOME'] = os.path.abspath('CoreNLP')
    #
    #     if required_words is not None:
    #         rws = set(required_words)
    #         filtered_captions = []
    #         for c in captions:
    #             cs = c.split()
    #             for rw in rws:
    #                 if rw in cs:
    #                     filtered_captions.append(c)
    #                     break
    #         captions = filtered_captions
    #
    #     with CoreNLPClient(annotators=['natlog', 'openie'], timeout=900_000, memory='4G', max_char_length=5_000_000) as client:
    #         triplets_str = []
    #
    #         batch_size = 1_000
    #         for b in range(int(np.ceil(len(captions) / batch_size))):
    #             print(f'{b * batch_size:6d}/{len(captions)}')
    #             caption_batch = captions[b:(b + 1) * batch_size]
    #             ann = client.annotate('. '.join(caption_batch))
    #             sentences = ann.sentence
    #             for i, s in enumerate(sentences):
    #                 try:
    #                     s, p, o = s.openieTriple[0].subject, s.openieTriple[0].relation, s.openieTriple[0].object
    #                     s = wn.morphy(s, wn.NOUN)
    #                     if s is not None:
    #                         if wn.morphy(p.split()[0], wn.VERB) == 'be':
    #                             p = ' '.join(p.split()[1:])
    #                         triplets_str.append([s, p, o])
    #                 except:
    #                     pass
    #     return triplets_str

    def get_interactions_for(self, hoi_ds: HoiDataset):
        # '_' -> ' '
        hoi_ds_action_index = {k.replace('_', ' '): v for k, v in hoi_ds.action_index.items()}
        hoi_ds_objects = [o.replace('_', ' ') for o in hoi_ds.objects]
        hoi_ds_object_index = {k.replace('_', ' '): v for k, v in hoi_ds.object_index.items()}

        # Subject mapping
        humans = set(self.human_classes)
        subj_mapping = np.full(len(self.objects), fill_value=-1, dtype=np.int)
        for s in humans:
            assert subj_mapping[s] == -1
            for t in [self.objects[s], 'person', 'human']:  # try specific one, then 'person', then 'human'
                if t in hoi_ds_object_index:
                    subj_mapping[s] = hoi_ds_object_index[t]
                    break

        # Predicate to action mapping
        pred_mapping = np.full(len(self.predicates), fill_value=-1, dtype=np.int)
        for i, pred in enumerate(self.predicates):
            pred_split = pred.split()

            if pred_split[0].startswith('text'):  # old WordNet doesn't have this
                verb_base_forms = ['text']
            else:
                # Using protected method to get all results instead of just the first one.
                verb_base_forms = wn._morphy(pred_split[0], wn.VERB, check_exceptions=True)
            if len(verb_base_forms) > 0:  # not a preposition
                for vbf in verb_base_forms:
                    verb_phrase_base_form = ' '.join([vbf] + pred_split[1:])
                    if verb_phrase_base_form in hoi_ds_action_index.keys():
                        pred_mapping[i] = hoi_ds_action_index[verb_phrase_base_form]
                        break
                else:
                    if 'drink' in verb_base_forms and len(pred_split) == 2 and pred_split[1] == 'from':  # drink_from -> drink_with
                        pred_mapping[i] = hoi_ds_action_index['drink with']

        # Object mapping
        fixes = {"ski's": 'skis',
                 'hairdryer': 'hair dryer',
                 'cellphone': 'cell phone'}
        obj_mapping = np.full(len(self.objects), fill_value=-1, dtype=np.int)
        for i, obj in enumerate(self.objects):
            obj = fixes.get(obj, obj)
            try:
                obj_mapping[i] = hoi_ds_object_index[obj]
            except KeyError:
                try:
                    obj_mapping[i] = hoi_ds_object_index[obj.split()[-1]]
                except KeyError:
                    try:
                        for j, o in enumerate(hoi_ds_objects):
                            if obj == o.split()[-1]:
                                obj_mapping[i] = j
                                break
                    except KeyError:
                        continue
        obj_mapping[np.array(self.human_classes)] = subj_mapping[np.array(self.human_classes)]

        # Relationship triplets to interactions
        relationships = np.unique(self.triplets, axis=0)
        mapped_relationships = np.stack([subj_mapping[relationships[:, 0]],
                                         pred_mapping[relationships[:, 1]],
                                         obj_mapping[relationships[:, 2]]],
                                        axis=1)
        relationships_to_interactions = np.unique(mapped_relationships[np.all(mapped_relationships >= 0, axis=1), 1:], axis=0)

        ts = [s.split('###') for s in sorted({'###'.join(t) for t in self.triplet_str})]
        inds = [i for i, s in enumerate(ts)
                if 'walk' in s[1]
                # if any([x in s[2].split() for x in ['cup', 'bottle']])
                ]
        strs = [' '.join(ts[i]) for i in inds]
        print('\n'.join(strs))

        # ts = [[f'{hoi_ds_actions[p] if p >= 0 else "-":20s}', f'{hoi_ds_objects[o] if o >= 0 else "-":20s}'] for s, p, o in mapped_relationships]
        # print('\n'.join([f'{p} {o}' for p, o in ts if 'text' in p if 'cell phone' in o]))

        return relationships_to_interactions


class HCVRD(ExtSource):
    def __init__(self):
        super().__init__()

    @property
    def human_classes(self) -> List[int]:
        return [self.object_index[o] for o in ['person', 'man', 'woman', 'boy', 'girl', 'child', 'kid', 'baby',
                                               'audience', 'catcher', 'carrier', 'classroom', 'couple', 'cowboy', 'crowd', 'driver', 'friend',
                                               'guard', 'little girl', 'player', 'rider', 'skateboarder', 'skater', 'skier', 'small child',
                                               'snowboarder', 'surfer', 'tennis player']]

    def _load(self):
        with open(os.path.join(cfg.data_root, 'HCVRD', 'final_data.json'), 'r') as f:
            d = json.load(f)  # {'im_id': [{'predicate', 'object', 'subject', 'obj_box', 'sub_box'}]}
        triplets_str = [[reldata['subject'], reldata['predicate'].strip(), reldata['object']] for imdata in d.values() for reldata in imdata]
        return triplets_str


class VG(ExtSource):
    def __init__(self):
        super().__init__()

    @property
    def human_classes(self) -> List[int]:
        return [self.object_index[o] for o in sorted({'person', 'man', 'woman', 'boy', 'girl', 'child', 'kid', 'baby', 'guy',
                                                      'audience', 'catcher', 'carrier', 'classroom', 'couple', 'cowboy', 'crowd', 'driver', 'friend',
                                                      'guard', 'little girl',
                                                      'player', 'rider', 'skateboarder', 'skater', 'skier', 'small child', 'snowboarder', 'surfer',
                                                      'tennis player'})]

    def _load(self):
        try:
            with open(os.path.join(cfg.cache_root, 'vg_parsed_rels.pkl'), 'rb') as f:
                triplets_str = pickle.load(f)
        except FileNotFoundError:
            d = json.load(open('data/VG/relationships.json', 'r'))
            triplets_str = []
            for imdata in d:
                rels = imdata['relationships']
                for r in rels:
                    s, p, o = r['subject'], r['predicate'], r['object']
                    so = [s, o]
                    for i, x in enumerate(so):
                        try:
                            so[i] = x['name']
                        except KeyError:
                            so[i] = x['names'][0]
                    s, o = so
                    s = wn.morphy(s, wn.NOUN)
                    if s is not None:
                        triplet = [x.strip().lower() for x in [s, p, o]]
                        if all(triplet):
                            triplets_str.append(triplet)

            with open(os.path.join('cache', 'vg_parsed_rels.pkl'), 'wb') as f:
                pickle.dump(triplets_str, f)
        return triplets_str


class VGCaptions(ExtSource):
    def __init__(self, required_words=None):
        self.required_words = required_words
        super().__init__()

    @property
    def human_classes(self) -> List[int]:
        return [self.object_index[o] for o in sorted({'person', 'man', 'woman', 'boy', 'girl', 'child', 'kid', 'baby', 'guy',
                                                      'audience', 'catcher', 'carrier', 'classroom', 'couple', 'cowboy', 'crowd', 'driver', 'friend',
                                                      'guard', 'little girl',
                                                      'player', 'rider', 'skateboarder', 'skater', 'skier', 'small child', 'snowboarder', 'surfer',
                                                      'tennis player'})]

    def _load(self):
        try:
            with open(os.path.join(cfg.cache_root, 'vg_triplets.pkl'), 'rb') as f:
                triplets_str = pickle.load(f)
        except FileNotFoundError:
            try:
                with open(os.path.join(cfg.cache_root, 'vg_region_descriptions.txt'), 'r') as f:
                    captions = [l.strip() for l in f.readlines()]
            except FileNotFoundError:
                assert False  # FIXME remove
                captions = json.load(open(os.path.join(cfg.data_root, 'region_descriptions.json'), 'r'))
                captions = [r['phrase'] for rd in captions for r in rd['regions']]
                with open(os.path.join(cfg.cache_root, 'vg_region_descriptions.txt'), 'w') as f:
                    f.write('\n'.join(captions))

            triplets_str = self._parse_captions(captions, required_words=self.required_words)

            with open(os.path.join(cfg.cache_root, 'vg_triplets.pkl'), 'wb') as f:
                pickle.dump(triplets_str, f)

        return triplets_str


class ActivityNetCaptions(ExtSource):
    def __init__(self, required_words=None):
        self.required_words = required_words
        super().__init__()

    @property
    def human_classes(self) -> List[int]:
        return [self.object_index[o] for o in sorted({'person', 'man', 'woman', 'boy', 'girl', 'child', 'kid', 'baby', 'guy',
                                                      'audience', 'catcher', 'carrier', 'classroom', 'couple', 'cowboy', 'crowd', 'driver', 'friend',
                                                      'guard', 'little girl',
                                                      'player', 'rider', 'skateboarder', 'skater', 'skier', 'small child', 'snowboarder', 'surfer',
                                                      'tennis player'})]

    def _load(self):
        try:
            with open(os.path.join(cfg.cache_root, 'anet_triplets.pkl'), 'rb') as f:
                triplets_str = pickle.load(f)
        except FileNotFoundError:
            d = json.load(open(os.path.join(cfg.data_root, 'VideoCaptions', 'train.json'), 'r'))
            captions = [s.strip(' .') for v in d.values() for s in v['sentences']]

            triplets_str = self._parse_captions(captions, required_words=self.required_words)

            with open(os.path.join(cfg.cache_root, 'anet_triplets.pkl'), 'wb') as f:
                pickle.dump(triplets_str, f)

        return triplets_str
