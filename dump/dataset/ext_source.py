from nltk.tag import pos_tag

import json
import os
import pickle
from typing import List, Dict, Tuple

import numpy as np
from nltk.corpus import wordnet as wn

from config import cfg
from lib.dataset.hoi_dataset import HoiDataset

HUMAN_CLASSES = {'person', 'man', 'woman', 'boy', 'girl', 'child', 'kid', 'baby', 'guy',  # Common
                 'audience', 'classroom', 'couple', 'crowd',  # Plural
                 # Sport
                 'catcher', 'player', 'rider', 'skateboarder', 'skater', 'skier', 'snowboarder', 'surfer', 'tennis player',
                 # Others
                 'friend', 'guard', 'small child', 'little girl', 'cowboy', 'carrier', 'driver', }


def _parse_sentences(sentences, required_words=None):
    from allennlp.predictors.predictor import Predictor

    if required_words is not None:
        rws = set(required_words)
        filtered_sentences = []
        for c in sentences:
            cs = c.split()
            for rw in rws:
                if rw in cs:
                    filtered_sentences.append(c)
                    break
        sentences = filtered_sentences

    dparser = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")

    triplets_str = []
    for i, c in enumerate(sentences):
        if i % 100 == 0:
            print(f'{i:6d}/{len(sentences)}')

        res = dparser.predict(sentence=c)
        s, p, o = None, res['hierplane_tree']['root']['word'], None
        for c in res['hierplane_tree']['root']['children']:
            if c['nodeType'] in {'nsubj', 'nn'}:
                s = c['word']
            elif c['nodeType'] == 'dobj':
                o = c['word']
            if s is not None and o is not None:
                triplets_str.append([s, p, o])
                break

    return triplets_str


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
        return [self.object_index[o] for o in sorted(HUMAN_CLASSES) if o in self.object_index.keys()]

    @property
    def triplet_str(self):
        return [(self.objects[s], self.predicates[p], self.objects[o]) for s, p, o in self.triplets]

    def _load(self):
        raise NotImplementedError

    def _normalise_triplets(self, triplets):
        norm_triplets = []
        for s, p, o in triplets:
            try:
                s = wn.morphy(s, wn.NOUN)
                if s is not None:
                    if wn.morphy(p.split()[0], wn.VERB) == 'be':
                        p = ' '.join(p.split()[1:])
                    p = p.strip()
                    if p:
                        norm_triplets.append([s, p, o])
            except:
                continue
        return norm_triplets

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

        # ts = [s.split('###') for s in sorted({'###'.join(t) for t in self.triplet_str})]
        # inds = [i for i, s in enumerate(ts)
        #         if 'walk' in s[1]
        #         # if any([x in s[2].split() for x in ['cup', 'bottle']])
        #         ]
        # strs = [' '.join(ts[i]) for i in inds]
        # print('\n'.join(strs))

        # ts = [[f'{hoi_ds_actions[p] if p >= 0 else "-":20s}', f'{hoi_ds_objects[o] if o >= 0 else "-":20s}'] for s, p, o in mapped_relationships]
        # print('\n'.join([f'{p} {o}' for p, o in ts if 'text' in p if 'cell phone' in o]))

        return relationships_to_interactions


class HCVRD(ExtSource):
    def __init__(self):
        super().__init__()

    @property
    def human_classes(self) -> List[int]:
        hcvrd_human_classes = {'person', 'man', 'woman', 'boy', 'girl', 'child', 'kid', 'baby',
                               'audience', 'catcher', 'carrier', 'classroom', 'couple', 'cowboy', 'crowd', 'driver', 'friend',
                               'guard', 'little girl', 'player', 'rider', 'skateboarder', 'skater', 'skier', 'small child',
                               'snowboarder', 'surfer', 'tennis player'}
        return [self.object_index[o] for o in sorted(hcvrd_human_classes)]

    def _load(self):
        with open(os.path.join(cfg.data_root, 'HCVRD', 'final_data.json'), 'r') as f:
            d = json.load(f)  # {'im_id': [{'predicate', 'object', 'subject', 'obj_box', 'sub_box'}]}
        triplets_str = [[reldata['subject'], reldata['predicate'].strip(), reldata['object']] for imdata in d.values() for reldata in imdata]
        return triplets_str


class VG(ExtSource):
    def __init__(self):
        super().__init__()

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

            triplets_str = self._normalise_triplets(_parse_sentences(captions, required_words=self.required_words))

            with open(os.path.join(cfg.cache_root, 'vg_triplets.pkl'), 'wb') as f:
                pickle.dump(triplets_str, f)

        return triplets_str


class ActivityNetCaptions(ExtSource):
    def __init__(self, required_words=None):
        self.required_words = required_words
        super().__init__()

    def _load(self):
        try:
            with open(os.path.join(cfg.cache_root, 'anet_triplets.pkl'), 'rb') as f:
                triplets_str = pickle.load(f)
        except FileNotFoundError:
            d = json.load(open(os.path.join(cfg.data_root, 'VideoCaptions', 'train.json'), 'r'))
            captions = [s.strip(' .') for v in d.values() for s in v['sentences']]

            parsed_triplets = _parse_sentences(captions, required_words=self.required_words)
            triplets_str = self._normalise_triplets(parsed_triplets)

            with open(os.path.join(cfg.cache_root, 'anet_triplets.pkl'), 'wb') as f:
                pickle.dump(triplets_str, f)

        triplets_str = [[s, p, o] for s, p, o in triplets_str if s and p and o]
        return triplets_str


class ImSitu(ExtSource):
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
                    EXAMPLE: Key: 'tatto'. Value:
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
        """
        self.nouns = self.verbs = None  # These will be initialised when superclass' constructor is called.
        super().__init__()

    @property
    def human_classes(self) -> List[int]:
        imsitu_human_classes = [s for s, p, o in self.triplet_str]
        return [self.object_index[o] for o in sorted(imsitu_human_classes)]

    def print_verb_entry(self, verb):
        print('ImSitu verb entry example:')
        for k, v in self.verbs[verb].items():
            if k != 'roles':
                print('%15s: %s' % (k, v))
            else:
                print('%15s:' % k)
                ln = max([len(r) for r in v.keys()]) + 1
                for r, d in v.items():
                    print(('%15s  - %-' + str(ln) + 's %s') % ('', r + ':', d))

    def _load(self):
        try:
            with open(os.path.join(cfg.cache_root, 'imsitu_triplets.pkl'), 'rb') as f:
                triplets_str = pickle.load(f)
        except FileNotFoundError:
            self.nouns, self.verbs, verb_instances_per_subj = self._load_raw_data()

            triplets_str = []
            for v, obj_per_subj in verb_instances_per_subj.items():
                assert not v.isupper()
                for wn_s, objs in obj_per_subj.items():
                    s = self.nouns[wn_s]['gloss'][0]
                    for wn_o in objs:
                        o = self.nouns[wn_o]['gloss'][0]
                        triplets_str.append([s, v, o])

            with open(os.path.join(cfg.cache_root, 'imsitu_triplets.pkl'), 'wb') as f:
                pickle.dump(triplets_str, f)
        return triplets_str

    @staticmethod
    def _load_raw_data() -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict]]:
        """
        train, val, test are of type dict(dict). Keys are image file names, values are dictionaries with the following keys:
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

        with open(os.path.join(data_dir, 'imsitu_space.json'), 'r') as f:
            domain = json.load(f)
        nouns = domain['nouns']
        verbs = {}
        for k, vdata in domain['verbs'].items():
            k = wn.morphy(k, wn.VERB)
            if k is not None:
                assert k not in verbs
                verbs[k] = vdata

        # Fix verbs
        verbs['ride']['abstract'] = verbs['ride']['abstract'].replace(' then ', ' the ')
        verbs['teach']['abstract'] = verbs['teach']['abstract'].replace(' to teach ', ' teaches ')
        for verb in verbs.keys():
            abstract = verbs[verb]['abstract']
            abstract = ' '.join([token.split('/')[0] for token in abstract.split()])  # 'an AGENT jumps over/through an OBSTACLE'
            if '(' in abstract:  # 'the SLIDER (when different from the AGENT) on'
                abstract = abstract.replace('(', '###').replace(')', '###')
                split = abstract.split('###')
                assert len(split) % 2 == 1  # Even number of parentheses
                abstract = ' '.join([x.strip() for x in split[::2]])
            verbs[verb]['abstract'] = abstract

        # Merge data
        with open(os.path.join(data_dir, 'train.json'), 'r') as f:
            train = json.load(f)
        with open(os.path.join(data_dir, 'dev.json'), 'r') as f:
            val = json.load(f)
        with open(os.path.join(data_dir, 'test.json'), 'r') as f:
            test = json.load(f)
        assert not set(train.keys()) & set(val.keys())
        assert not set(train.keys()) & set(test.keys())
        assert not set(val.keys()) & set(test.keys())
        data = {k: v for split in [train, val, test] for k, v in split.items()}

        # Get verb instances
        verb_instances_per_subj = {}
        for vdata in data.values():
            verb, frame_list = vdata['verb'], vdata['frames']
            verb_base_form = wn.morphy(verb, wn.VERB)
            if verb_base_form and len(verbs[verb_base_form]['order']) > 1:
                subj_key = verbs[verb_base_form]['order'][0]
                obj_key = verbs[verb_base_form]['order'][1]
                # if obj_key == 'agentparts':  # FIXME
                #     obj_key = obj_key[:-1].upper()

                abstract = verbs[verb_base_form]['abstract']
                abstract_tokens = [t.split("'")[0] for t in abstract.replace('’', "'").strip(' .').split()]
                try:
                    obj_idx = abstract_tokens.index(obj_key.upper())
                    for i, t in enumerate(abstract_tokens[:obj_idx - 2]):  # -2 because if there is a preposition it has to be at most in position -1
                        if wn.morphy(t, wn.VERB) == verb_base_form:
                            possible_preposition = abstract_tokens[i+1]
                            tagged_token = pos_tag([possible_preposition], tagset='universal')
                            if tagged_token[0][1] == 'ADP':
                                preposition = tagged_token[0][0]
                                verb_base_form = f'{verb_base_form} {preposition}'
                            break
                except ValueError:
                    pass

                for frame in frame_list:
                    concrete_subj = frame[subj_key]
                    concrete_obj = frame[obj_key]
                    if concrete_subj and concrete_obj:
                        verb_instances_per_subj.setdefault(verb_base_form, {}).setdefault(concrete_subj, set()).add(concrete_obj)

        return nouns, verbs, verb_instances_per_subj
