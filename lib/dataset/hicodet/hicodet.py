import os
import pickle
from typing import Dict, List

import numpy as np
from scipy.io import loadmat

from config import cfg
from lib.dataset.utils import Splits
from lib.dataset.hoi_dataset import HoiDataset


class HicoDetImData:
    def __init__(self, filename, boxes, box_classes, hois, wnet_actions):
        self.filename = filename  # type: str
        self.boxes = boxes  # type: np.ndarray  # Nx4 int
        self.box_classes = box_classes  # type: np.ndarray  # N int
        self.hois = hois  # type: np.ndarray  # Mx3, [hum_idx, action_id, obj_idx] int
        self.wnet_actions = wnet_actions  # type: List[str]


class HicoDet(HoiDataset):
    def __init__(self):
        driver = HicoDetDriver()  # type: HicoDetDriver
        null_action = driver.null_interaction
        objects = sorted(set([inter['obj'] for inter in driver.interaction_list]))
        actions = list(driver.action_dict.keys())
        interactions_classes = [[inter['act'], inter['obj']] for inter in driver.interaction_list]

        super().__init__(object_classes=objects, action_classes=actions, null_action=null_action, interactions_classes=interactions_classes)
        self.driver = driver  # type: HicoDetDriver
        self.split_data = {Splits.TRAIN: self.compute_annotations(Splits.TRAIN),
                           Splits.TEST: self.compute_annotations(Splits.TEST)
                           }  # type: Dict[Splits: List[HicoDetImData]]

        self.split_non_empty_image_ids = {s: [i for i, im_data in enumerate(self.split_data[s]) if im_data.boxes.size > 0]
                                          for s in [Splits.TRAIN, Splits.TEST]}  # empty image = doesn't have annotations

    @property
    def human_class(self) -> int:
        return self.object_index['person']

    @property
    def num_objects(self):
        return len(self.objects)

    @property
    def num_actions(self):
        return len(self.actions)

    @property
    def num_interactions(self):
        return self.interactions.shape[0]

    def get_img_dir(self, split):
        return self.driver.split_img_dir[split]

    def get_triplets(self, split, which='all'):
        choices = ['all', 'rare', 'non rare']
        if which not in choices:
            raise ValueError(f'Incorrect value "{which}". Choose from f{choices}.')
        data = self.split_data[split]
        hoi_triplets = np.concatenate([np.stack([imd.box_classes[imd.hois[:, 0]],
                                                 imd.hois[:, 1],
                                                 imd.box_classes[imd.hois[:, 2]]], axis=1) for imd in data], axis=0)
        assert np.all(hoi_triplets[:, 0] == self.human_class)
        if which != choices[0]:
            u_hoi_triplets, counts = np.unique(hoi_triplets, axis=0, return_counts=True)
            if which == choices[1]:
                hoi_triplets = u_hoi_triplets[counts < 10]
            else:
                assert which == choices[2]
                hoi_triplets = u_hoi_triplets[counts >= 10]
        return hoi_triplets

    def compute_annotations(self, split) -> List[HicoDetImData]:
        annotations = self.driver.split_annotations[split if split == Splits.TEST else Splits.TRAIN]
        split_data = []
        for i, img_ann in enumerate(annotations):
            im_hum_boxes, im_obj_boxes, im_obj_box_classes, im_interactions, im_wn_actions = [], [], [], [], []
            for inter in img_ann['interactions']:
                inter_id = inter['id']
                if not inter['invis']:
                    curr_num_hum_boxes = int(sum([b.shape[0] for b in im_hum_boxes]))
                    curr_num_obj_boxes = int(sum([b.shape[0] for b in im_obj_boxes]))

                    # Interaction
                    act_class = self.interactions[inter_id][0]
                    new_inters = inter['conn']
                    num_new_inters = new_inters.shape[0]
                    new_inters = np.stack([new_inters[:, 0] + curr_num_hum_boxes,
                                           np.full(num_new_inters, fill_value=act_class, dtype=np.int),
                                           new_inters[:, 1] + curr_num_obj_boxes
                                           ], axis=1)
                    im_interactions.append(new_inters)
                    im_wn_actions += [self.driver.interaction_list[inter_id]['act_wid'] for _ in range(num_new_inters)]

                    # Human
                    im_hum_boxes.append(inter['hum_bbox'])

                    # Object
                    obj_boxes = inter['obj_bbox']
                    im_obj_boxes.append(obj_boxes)
                    obj_class = self.interactions[inter_id][1]
                    im_obj_box_classes.append(np.full(obj_boxes.shape[0], fill_value=obj_class, dtype=np.int))

            if im_hum_boxes:
                assert im_obj_boxes
                assert im_obj_box_classes
                assert im_interactions
                im_hum_boxes, inv_ind = np.unique(np.concatenate(im_hum_boxes, axis=0), axis=0, return_inverse=True)
                num_hum_boxes = im_hum_boxes.shape[0]

                im_obj_boxes = np.concatenate(im_obj_boxes)
                im_obj_box_classes = np.concatenate(im_obj_box_classes)

                im_interactions = np.concatenate(im_interactions)
                im_interactions[:, 0] = np.array([inv_ind[h] for h in im_interactions[:, 0]], dtype=np.int)
                im_interactions[:, 2] += num_hum_boxes

                im_boxes = np.concatenate([im_hum_boxes, im_obj_boxes], axis=0)
                im_box_classes = np.concatenate([np.full(num_hum_boxes, fill_value=self.human_class, dtype=np.int), im_obj_box_classes])
            else:
                im_boxes = np.empty((0, 4), dtype=np.int)
                im_box_classes = np.empty(0, dtype=np.int)
                im_interactions = np.empty((0, 3), dtype=np.int)
            split_data.append(HicoDetImData(filename=img_ann['file'], boxes=im_boxes, box_classes=im_box_classes, hois=im_interactions,
                                            wnet_actions=im_wn_actions))

        return split_data


class HicoDetDriver:
    def __init__(self):
        """
        Relevant class attributes:
            - null_interaction: the name of the null interaction
            - wn_action_dict [dict]: The 119 WordNet entries for all actions. Keys are wordnets IDs and each element contains:
                - 'wname' [str]: The name of the wordnet entry this actions refers to. It is in the form VERB.v.NUM, where VERB is the verb
                    describing the action and NUM is an index used to disambiguate between homonyms.
                - 'id' [int]: A number I have not understood the use of.
                - 'count' [int]: Another number I have not understood the use of.
                - 'syn' [list]: Set of synonyms
                - 'def' [str]: A definition
                - 'ex' [str]: An example (sometimes not provided)
                EXAMPLE: key: v00007012, entry:
                    {'id': 1, 'wname': 'blow.v.01', 'count': 6, 'syn': ['blow'], 'def': 'exhale hard', 'ex': 'blow on the soup to cool it down'}
            - action_dict [dict]: The 117 possible actions, including a null one. They are fewer than the entries in the WordNet dictionary
                because some action can have different meaning and thus two different WordNet entries. Keys are verbs in the base form and
                entries consist of:
                    - 'ing' [str]: -ing form of the verb (unchanged for the null one).
                    - 'wn_ids' [list(str)]: The WordNet IDs (AKA keys in `wn_action_dict`) corresponding to this verb (empty for the null one).
            - interaction_list [list(dict)]: The 600 interactions in HICO-DET. Each element consists of:
                - 'obj' [str]: The name of the object of the action (i.e., the target).
                - 'act' [str]: The verb describing the action (key in `action_dict`).
                - 'act_wid' [str]: The WordNet ID of the action (key in `wn_action_dict`), or None for the null interaction.
            - split_data [dict(dict)]: One entry per split, with keys in `Splits`. Each entry is a dictionary with the following items:
                - 'img_dir' [str]: Path to the folder containing the images
                - 'annotations' [list(dict)]: Annotations for each image, thus structured:
                    - 'file' [str]: The file name
                    - 'orig_img_size' [array]: Image size expressed in [width, height, depth]
                    - 'interactions' [list(dict)]: Each entry has:
                            - 'id' [int]: The id of the interaction in `interaction_list`.
                            - 'invis' [bool]: Whether the interaction is invisible or not. It does NOT necesserily mean that it is not in the image.
                        If 'invis' is False then there are three more fields:
                            - 'hum_bbox' [array]: Hx4 matrix of (x1, y1, x2, y2) coordinates for each bounding box belonging to a human.
                            - 'obj_bbox' [array]: Ox4 matrix of (x1, y1, x2, y2) coordinates for each bounding box belonging to an object.
                            - 'conn' [array]: Cx2 with a pair of human-object indices for each interaction
                Other entries might be added to this dictionary for caching reasons.
        """

        self.data_dir = os.path.join(cfg.data_root, 'HICO-DET')
        self.path_pickle_annotation_file = os.path.join(self.data_dir, 'annotations.pkl')
        self.null_interaction = '__no_interaction__'

        train_annotations, test_annotations, interaction_list, wn_act_dict, act_dict = self.load_annotations()
        self.split_img_dir = {Splits.TRAIN: os.path.join(self.data_dir, 'images', 'train2015'),
                              Splits.TEST: os.path.join(self.data_dir, 'images', 'test2015')}
        self.split_annotations = {Splits.TRAIN: train_annotations, Splits.TEST: test_annotations}
        self.interaction_list = interaction_list
        self.wn_action_dict = wn_act_dict
        self.action_dict = act_dict

    def load_annotations(self):
        def _parse_split(_split):
            # The many "-1"s are due to original values being suited for MATLAB.
            _annotations = []
            for _src_ann in src_anns['bbox_%s' % _split.value]:
                _ann = {'file': _src_ann[0],
                        'orig_img_size': np.array([int(_src_ann[1][field]) for field in ['width', 'height', 'depth']], dtype=np.int),
                        'interactions': []}
                for _inter in np.atleast_1d(_src_ann[2]):
                    _new_inter = {
                        'id': int(_inter['id']) - 1,
                        'invis': bool(_inter['invis']),
                    }
                    if not _new_inter['invis']:
                        _new_inter['hum_bbox'] = np.atleast_2d(np.array([_inter['bboxhuman'][c] - 1 for c in ['x1', 'y1', 'x2', 'y2']],
                                                                        dtype=np.int).T)
                        _new_inter['obj_bbox'] = np.atleast_2d(np.array([_inter['bboxobject'][c] - 1 for c in ['x1', 'y1', 'x2', 'y2']],
                                                                        dtype=np.int).T)
                        _new_inter['conn'] = np.atleast_2d(np.array([coord - 1 for coord in _inter['connection']], dtype=np.int))
                    _ann['interactions'].append(_new_inter)
                _annotations.append(_ann)
            return _annotations

        try:
            with open(self.path_pickle_annotation_file, 'rb') as f:
                d = pickle.load(f)
                train_annotations = d[Splits.TRAIN.value]
                test_annotations = d[Splits.TEST.value]
                interaction_list = d['interaction_list']
                wn_act_dict = d['wn_act_dict']
                act_dict = d['act_dict']
        except FileNotFoundError:
            src_anns = loadmat(os.path.join(self.data_dir, 'anno_bbox.mat'), squeeze_me=True)

            train_annotations = _parse_split(_split=Splits.TRAIN)
            test_annotations = _parse_split(_split=Splits.TEST)

            interaction_list, wn_act_dict, act_dict = self.parse_interaction_list(src_anns['list_action'])

            with open(self.path_pickle_annotation_file, 'wb') as f:
                pickle.dump({Splits.TRAIN.value: train_annotations,
                             Splits.TEST.value: test_annotations,
                             'interaction_list': interaction_list,
                             'wn_act_dict': wn_act_dict,
                             'act_dict': act_dict,
                             }, f)

        # Substitute 'no_interaction' with the specified null interaction string, if needed.
        act_dict[self.null_interaction] = act_dict.get('no_interaction', self.null_interaction)
        del act_dict['no_interaction']
        act_dict = {k: act_dict[k] for k in sorted(act_dict.keys())}
        for inter in interaction_list:
            if inter['act'] == 'no_interaction':
                inter['act'] = self.null_interaction
            if inter['obj'] == 'hair_drier':
                inter['obj'] = 'hair_dryer'

        return train_annotations, test_annotations, interaction_list, wn_act_dict, act_dict

    @staticmethod
    def parse_interaction_list(src_interaction_list):
        wact_dict = {}
        interaction_list = []
        act_dict = {}

        for i, interaction_ann in enumerate(src_interaction_list):
            fields = interaction_ann[-2].dtype.fields
            act_wann = {}
            act_wid = None
            if fields is None:  # Null interaction
                for j, s in enumerate(interaction_ann):
                    if j < 3:
                        if j > 0:
                            assert s == 'no_interaction'
                        assert isinstance(s, str)
                    else:
                        assert s.size == 0
            else:
                for f in fields:
                    fvalue = str(interaction_ann[-2][f])
                    try:
                        fvalue = int(fvalue)
                    except ValueError:
                        pass

                    if f == 'name':
                        act_wann['wname'] = fvalue
                    elif f == 'wid':
                        act_wid = fvalue
                    elif f == 'syn':
                        act_wann[f] = list(set(fvalue.split(' ')))
                    elif f == 'ex':
                        act_wann[f] = fvalue if fvalue != '[]' else ''
                    else:
                        act_wann[f] = fvalue

                # Add to the wordnet action dictionary
                assert wact_dict.setdefault(act_wid, act_wann) == act_wann, '\n%s\n%s' % (wact_dict[act_wid], act_wann)

            assert 'name' not in act_wann

            # Add to the action dictionary
            act, act_ing = interaction_ann[1], interaction_ann[2]
            d_act = act_dict.setdefault(act, {'ing': act_ing, 'wn_ids': []})
            assert d_act['ing'] == act_ing
            if act_wid is not None:
                act_dict[act]['wn_ids'] = sorted(set(act_dict[act]['wn_ids'] + [act_wid]))

            # Add to the interaction list
            new_action_ann = {'obj': interaction_ann[0], 'act': act, 'act_wid': act_wid}
            interaction_list.append(new_action_ann)

        # Sort
        wact_dict = {k: wact_dict[k] for k in sorted(wact_dict.keys())}
        act_dict = {k: act_dict[k] for k in sorted(act_dict.keys())}

        return interaction_list, wact_dict, act_dict
