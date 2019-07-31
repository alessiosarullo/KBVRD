import os
import pickle

import numpy as np
from scipy.io import loadmat

from lib.dataset.utils import Splits
from typing import Dict, List


class HicoDetImData:
    def __init__(self, filename, boxes, box_classes, hois, wnet_actions):
        self.filename = filename
        self.boxes = boxes
        self.box_classes = box_classes
        self.hois = hois
        self.wnet_actions = wnet_actions


class HicoDet:
    def __init__(self):
        self.driver = HicoDetDriver()  # type: HicoDetDriver
        self.null_interaction = self.driver.null_interaction

        # Objects
        self.objects = sorted(set([inter['obj'] for inter in self.driver.interaction_list]))
        self.object_index = {obj: i for i, obj in enumerate(self.objects)}

        # Predicates
        self.predicates = list(self.driver.predicate_dict.keys())
        self.predicate_index = {pred: i for i, pred in enumerate(self.predicates)}
        assert self.predicate_index[self.null_interaction] == 0

        # Interactions
        self.interactions = np.array([[self.predicate_index[inter['pred']], self.object_index[inter['obj']]]
                                      for inter in self.driver.interaction_list])  # 600 x [p, o]
        self.op_pair_to_interaction = np.full([self.num_object_classes, self.num_actions], fill_value=-1, dtype=np.int)
        self.op_pair_to_interaction[self.interactions[:, 1], self.interactions[:, 0]] = np.arange(self.num_interactions)

        # Data
        self.split_data = {Splits.TRAIN: self.compute_annotations(Splits.TRAIN),
                           Splits.TEST: self.compute_annotations(Splits.TEST)
                           }  # type: Dict[Splits: List[HicoDetImData]]

    @property
    def human_class(self) -> int:
        return self.object_index['person']

    @property
    def num_object_classes(self):
        return len(self.objects)

    @property
    def num_actions(self):
        return len(self.predicates)

    @property
    def num_interactions(self):
        return self.interactions.shape[0]

    def get_img_dir(self, split):
        return self.driver.split_img_dir[split]

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
                    pred_class = self.interactions[inter_id][0]
                    new_inters = inter['conn']
                    num_new_inters = new_inters.shape[0]
                    new_inters = np.stack([new_inters[:, 0] + curr_num_hum_boxes,
                                           np.full(num_new_inters, fill_value=pred_class, dtype=np.int),
                                           new_inters[:, 1] + curr_num_obj_boxes
                                           ], axis=1)
                    im_interactions.append(new_inters)
                    im_wn_actions += [self.driver.interaction_list[inter_id]['pred_wid'] for _ in range(num_new_inters)]

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
            - wn_predicate_dict [dict]: The 119 WordNet entries for all predicates. Keys are wordnets IDs and each element contains:
                - 'wname' [str]: The name of the wordnet entry this actions refers to. It is in the form VERB.v.NUM, where VERB is the verb
                    describing the action and NUM is an index used to disambiguate between homonyms.
                - 'id' [int]: A number I have not understood the use of.
                - 'count' [int]: Another number I have not understood the use of.
                - 'syn' [list]: Set of synonyms
                - 'def' [str]: A definition
                - 'ex' [str]: An example (sometimes not provided)
                EXAMPLE: key: v00007012, entry:
                    {'id': 1, 'wname': 'blow.v.01', 'count': 6, 'syn': ['blow'], 'def': 'exhale hard', 'ex': 'blow on the soup to cool it down'}
            - predicate_dict [dict]: The 117 possible predicates, including a null one. They are fewer than the entries in the WordNet dictionary
                because some predicate can have different meaning and thus two different WordNet entries. Keys are verbs in the base form and
                entries consist of:
                    - 'ing' [str]: -ing form of the verb (unchanged for the null one).
                    - 'wn_ids' [list(str)]: The WordNet IDs (AKA keys in `wn_predicate_dict`) corresponding to this verb (empty for the null one).
            - interaction_list [list(dict)]: The 600 interactions in HICO-DET. Each element consists of:
                - 'obj' [str]: The name of the object of the action (i.e., the target).
                - 'pred' [str]: The verb describing the action (key in `predicate_dict`).
                - 'pred_wid' [str]: The WordNet ID of the action (key in `wn_predicate_dict`), or None for the null interaction.
            - split_data [dict(dict)]: One entry per split, with keys in `Splits`. Each entry is a dictionary with the following items:
                - 'img_dir' [str]: Path to the folder containing the images
                - 'annotations' [list(dict)]: Annotations for each image, thus structured:
                    - 'file' [str]: The file name
                    - 'img_size' [array]: Image size expressed in [width, height, depth]
                    - 'interactions' [list(dict)]: Each entry has:
                            - 'id' [int]: The id of the interaction in `interaction_list`.
                            - 'invis' [bool]: Whether the interaction is invisible or not. It does NOT necesserily mean that it is not in the image.
                        If 'invis' is False then there are three more fields:
                            - 'hum_bbox' [array]: Hx4 matrix of (x1, y1, x2, y2) coordinates for each bounding box belonging to a human.
                            - 'obj_bbox' [array]: Ox4 matrix of (x1, y1, x2, y2) coordinates for each bounding box belonging to an object.
                            - 'conn' [array]: Cx2 with a pair of human-object indices for each interaction
                Other entries might be added to this dictionary for caching reasons.
        """
        # TODO what are the 'id' and 'count' field for?

        self.data_dir = os.path.join('data', 'HICO-DET')
        self.path_pickle_annotation_file = os.path.join(self.data_dir, 'annotations.pkl')
        self.null_interaction = '__no_interaction__'

        train_annotations, test_annotations, interaction_list, wn_pred_dict, pred_dict = self.load_annotations()
        self.split_img_dir = {Splits.TRAIN: os.path.join(self.data_dir, 'images', 'train2015'),
                              Splits.TEST: os.path.join(self.data_dir, 'images', 'test2015')}
        self.split_annotations = {Splits.TRAIN: train_annotations, Splits.TEST: test_annotations}
        self.interaction_list = interaction_list
        self.wn_predicate_dict = wn_pred_dict
        self.predicate_dict = pred_dict

    def load_annotations(self):
        def _parse_split(_split):
            # The many "-1"s are due to original values being suited for MATLAB.
            _annotations = []
            for _src_ann in src_anns['bbox_%s' % _split.value]:
                _ann = {'file': _src_ann[0],
                        'img_size': np.array([int(_src_ann[1][field]) for field in ['width', 'height', 'depth']], dtype=np.int),
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
                wn_pred_dict = d['wn_pred_dict']
                pred_dict = d['pred_dict']
        except FileNotFoundError:
            src_anns = loadmat(os.path.join(self.data_dir, 'anno_bbox.mat'), squeeze_me=True)

            train_annotations = _parse_split(_split=Splits.TRAIN)
            test_annotations = _parse_split(_split=Splits.TEST)

            interaction_list, wn_pred_dict, pred_dict = self.parse_interaction_list(src_anns['list_action'])

            with open(self.path_pickle_annotation_file, 'wb') as f:
                pickle.dump({Splits.TRAIN.value: train_annotations,
                             Splits.TEST.value: test_annotations,
                             'interaction_list': interaction_list,
                             'wn_pred_dict': wn_pred_dict,
                             'pred_dict': pred_dict,
                             }, f)

        # Substitute 'no_interaction' with the specified null interaction string, if needed.
        pred_dict[self.null_interaction] = pred_dict.get('no_interaction', self.null_interaction)
        del pred_dict['no_interaction']
        pred_dict = {k: pred_dict[k] for k in sorted(pred_dict.keys())}
        for inter in interaction_list:
            if inter['pred'] == 'no_interaction':
                inter['pred'] = self.null_interaction
            if inter['obj'] == 'hair_drier':
                inter['obj'] = 'hair_dryer'

        return train_annotations, test_annotations, interaction_list, wn_pred_dict, pred_dict

    @staticmethod
    def parse_interaction_list(src_interaction_list):
        wpred_dict = {}
        interaction_list = []
        pred_dict = {}

        for i, interaction_ann in enumerate(src_interaction_list):
            fields = interaction_ann[-2].dtype.fields
            pred_wann = {}
            pred_wid = None
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
                        pred_wann['wname'] = fvalue
                    elif f == 'wid':
                        pred_wid = fvalue
                    elif f == 'syn':
                        pred_wann[f] = list(set(fvalue.split(' ')))
                    elif f == 'ex':
                        pred_wann[f] = fvalue if fvalue != '[]' else ''
                    else:
                        pred_wann[f] = fvalue

                # Add to the wordnet predicate dictionary
                assert wpred_dict.setdefault(pred_wid, pred_wann) == pred_wann, '\n%s\n%s' % (wpred_dict[pred_wid], pred_wann)

            assert 'name' not in pred_wann

            # Add to the predicate dictionary
            pred, pred_ing = interaction_ann[1], interaction_ann[2]
            d_pred = pred_dict.setdefault(pred, {'ing': pred_ing, 'wn_ids': []})
            assert d_pred['ing'] == pred_ing
            if pred_wid is not None:
                pred_dict[pred]['wn_ids'] = sorted(set(pred_dict[pred]['wn_ids'] + [pred_wid]))

            # Add to the interaction list
            new_action_ann = {'obj': interaction_ann[0], 'pred': pred, 'pred_wid': pred_wid}
            interaction_list.append(new_action_ann)

        # Sort
        wpred_dict = {k: wpred_dict[k] for k in sorted(wpred_dict.keys())}
        pred_dict = {k: pred_dict[k] for k in sorted(pred_dict.keys())}

        return interaction_list, wpred_dict, pred_dict
