import os
import pickle

import numpy as np
from scipy.io import loadmat

from lib.dataset.utils import Splits


class Hico:
    def __init__(self):
        self.driver = HicoDriver()  # type: HicoDriver
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
        self.op_pair_to_interaction = np.full([self.num_object_classes, self.num_predicates], fill_value=-1, dtype=np.int)
        self.op_pair_to_interaction[self.interactions[:, 1], self.interactions[:, 0]] = np.arange(self.num_interactions)

        # Data
        train_annotations = self.driver.split_annotations[Splits.TRAIN]
        train_annotations[np.isnan(train_annotations)] = 0
        test_annotations = self.driver.split_annotations[Splits.TEST]
        test_annotations[np.isnan(test_annotations)] = 0
        self.split_annotations = {Splits.TRAIN: train_annotations, Splits.TEST: test_annotations}
        self.split_filenames = self.driver.split_filenames

    @property
    def human_class(self) -> int:
        return self.object_index['person']

    @property
    def num_object_classes(self):
        return len(self.objects)

    @property
    def num_predicates(self):
        return len(self.predicates)

    @property
    def num_interactions(self):
        return self.interactions.shape[0]

    def get_img_dir(self, split):
        return self.driver.split_img_dir[split]


class HicoDriver:
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

        self.data_dir = os.path.join('data', 'HICO')
        self.path_pickle_annotation_file = os.path.join(self.data_dir, 'annotations.pkl')
        self.null_interaction = '__no_interaction__'

        train_annotations, train_fns, test_annotations, test_fns, interaction_list, wn_pred_dict, pred_dict = self.load_annotations()
        self.split_img_dir = {Splits.TRAIN: os.path.join(self.data_dir, 'images', 'train2015'),
                              Splits.TEST: os.path.join(self.data_dir, 'images', 'test2015')}
        self.split_annotations = {Splits.TRAIN: train_annotations, Splits.TEST: test_annotations}
        self.split_filenames = {Splits.TRAIN: train_fns, Splits.TEST: test_fns}
        self.interaction_list = interaction_list
        self.wn_predicate_dict = wn_pred_dict
        self.predicate_dict = pred_dict

    def load_annotations(self):
        try:
            with open(self.path_pickle_annotation_file, 'rb') as f:
                d = pickle.load(f)
                train_annotations = d[f'{Splits.TRAIN.value}_anno']
                train_fns = d[f'{Splits.TRAIN.value}_fn']
                test_annotations = d[f'{Splits.TEST.value}_anno']
                test_fns = d[f'{Splits.TEST.value}_fn']
                interaction_list = d['interaction_list']
                wn_pred_dict = d['wn_pred_dict']
                pred_dict = d['pred_dict']
        except FileNotFoundError:
            # 'anno_train': 600 x 38118 matrix. Associates to each training set images action labels. Specifically, cell (i,j) can contain one
            #       of the four values -1, 0, 1 or NaN according to whether action i is a hard negative, a soft negative/positive,
            #       a hard positive or unknown in image j.
            # 'anno_test': 600 x 9658 matrix. Same format for the training set one.
            # 'list_train' and 'list_set' are respectively 38118- and 9658- dimensional vectors of file names.
            src_anns = loadmat(os.path.join(self.data_dir, 'anno.mat'), squeeze_me=True)

            train_annotations = src_anns['anno_train'].T
            train_fns = [fn for fn in src_anns['list_train']]
            test_annotations = src_anns['anno_test'].T
            test_fns = [fn for fn in src_anns['list_test']]
            interaction_list, wn_pred_dict, pred_dict = self.parse_interaction_list(src_anns['list_action'])

            with open(self.path_pickle_annotation_file, 'wb') as f:
                pickle.dump({f'{Splits.TRAIN.value}_anno': train_annotations,
                             f'{Splits.TRAIN.value}_fn': train_fns,
                             f'{Splits.TEST.value}_anno': test_annotations,
                             f'{Splits.TEST.value}_fn': test_fns,
                             'interaction_list': interaction_list,
                             'wn_pred_dict': wn_pred_dict,
                             'pred_dict': pred_dict,
                             }, f)

        assert train_annotations.shape[0] == len(train_fns)
        assert test_annotations.shape[0] == len(test_fns)

        # Substitute 'no_interaction' with the specified null interaction string, if needed.
        pred_dict[self.null_interaction] = pred_dict.get('no_interaction', self.null_interaction)
        del pred_dict['no_interaction']
        pred_dict = {k: pred_dict[k] for k in sorted(pred_dict.keys())}
        for inter in interaction_list:
            if inter['pred'] == 'no_interaction':
                inter['pred'] = self.null_interaction
            if inter['obj'] == 'hair_drier':
                inter['obj'] = 'hair_dryer'

        return train_annotations, train_fns, test_annotations, test_fns, interaction_list, wn_pred_dict, pred_dict

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


if __name__ == '__main__':
    Hico()
