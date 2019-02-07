import os
import pickle

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.io import loadmat
from utils.data import Splits
from typing import List, Dict


class HicoDet:
    def __init__(self):
        """
        Class attributes:
            - wn_predicate_dict: [dict] The 119 WordNet entries for all predicates. Keys are wordnets IDs and each element contains:
                - 'wname': [str] The name of the wordnet entry this actions refers to. It is in the form VERB.v.NUM, where VERB is the verb
                    describing the action and NUM is an index used to disambiguate between homonyms.
                - 'id': [int] A number I have not understood the use of.
                - 'count': [int] Another number I have not understood the use of.
                - 'syn': [list] Set of synonyms
                - 'def': [str] A definition
                - 'ex': [str] An example (sometimes not provided)
                EXAMPLE: key: v00007012, entry:
                    {'id': 1, 'wname': 'blow.v.01', 'count': 6, 'syn': ['blow'], 'def': 'exhale hard', 'ex': 'blow on the soup to cool it down'}
            - predicate_dict: [dict] The 117 possible predicates, including a null one. They are less than the entries in the WordNet dictionary
                because some predicate can have different meaning and thus two different WordNet entries. Keys are verbs in the base form and
                entries consist of:
                    - 'ing': [str] -ing form of the verb ('no_interaction' for the null one).
                    - 'wn_ids': [list(str)] The WordNet IDs (AKA keys in `wn_predicate_dict`) corresponding to this verb (empty for the null one).
            - interaction_list: [list(dict)] The 600 interactions in HICO-DET. Each element consists of:
                - 'obj': [str] The name of the object of the action (i.e., the target).
                - 'pred': [str] The verb describing the action (key in `predicate_dict`).
                - 'pred_wid': [str] The WordNet ID of the action (key in `wn_predicate_dict`).
            - split_data: [dict(dict)] One entry per split, with keys in `Splits`. Each entry is a dictionary with the following items:
                - 'img_dir': [str] Path to the folder containing the images
                - 'annotations': [list(dict)] Annotations for each image, thus structured:
                    - 'file': [str] The file name
                    - 'img_size': [array] Image size expressed in [width, height, depth]
                    - 'interactions': [list(dict)] Each entry has:
                            - 'id': [int] The id of the interaction in `interaction_list`.
                            - 'invis': [bool] Whether the interaction is invisible or not. It does NOT necesserily mean that it is not in the image.
                        If 'invis' is False then there are three more fields:
                            - 'hum_bbox': [array] Hx4 matrix of (x1, y1, x2, y2) coordinates for each bounding box belonging to a human.
                            - 'obj_bbox': [array] Ox4 matrix of (x1, y1, x2, y2) coordinates for each bounding box belonging to an object.
                            - 'conn': [array] Cx2 with a pair of human-object indices for each interaction
                Other entries might be added to this dictionary for caching reasons.
        """
        # FIXME what are the 'id' and 'count' field for?

        self.data_dir = os.path.join('data', 'HICO-DET')
        self.split_data = {Splits.TRAIN: {'img_dir': os.path.join(self.data_dir, 'images', 'train2015')
                                          },
                           Splits.TEST: {'img_dir': os.path.join(self.data_dir, 'images', 'test2015')
                                         },
                           }
        self.path_pickle_annotation_file = os.path.join(self.data_dir, 'annotations.pkl')

        train_annotations, test_annotations, interaction_list, wn_pred_dict, pred_dict = self.load_annotations(use_hico_det=True)
        self.split_data[Splits.TRAIN]['annotations'] = train_annotations
        self.split_data[Splits.TEST]['annotations'] = test_annotations
        self._interaction_list = interaction_list
        self.wn_predicate_dict = wn_pred_dict
        self.predicate_dict = pred_dict

        # Derived
        self._objects = sorted(set([inter['obj'] for inter in self.interactions]))
        self._predicates = list(self.predicate_dict.keys())
        self.obj_class_index = {obj: i for i, obj in enumerate(self.objects)}
        self.pred_index = {pred: i for i, pred in enumerate(self.predicates)}

        # Statistics
        self.compute_stats()

        # # Sanity check
        # self.sanity_check()

        # HICO_train2015_00007266.jpg: kid eating apple. Index in `self.train_annotations`: 7257
        # self.show(7257)
        # self.show(0)

    def show(self, im_id, split=Splits.TRAIN):
        annotations = self.split_data[split]['annotations']
        im_dir = self.split_data[split]['img_dir']

        ann = annotations[im_id]

        img = np.array(Image.open(os.path.join(im_dir, ann['file'])))
        plt.imshow(img)
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        print(ann['img_size'])
        inter = ann['interactions'][1]
        print(self.interactions[inter['id']]['pred'], self.interactions[inter['id']]['obj'])
        for field in inter.keys():
            print(field, inter[field])
        for k in ['hum_bbox', 'obj_bbox']:
            bbox = inter[k][0, :]
            plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]),
                                              bbox[2] - bbox[0],
                                              bbox[3] - bbox[1],
                                              fill=False,
                                              edgecolor='blue' if k == 'hum_bbox' else 'green'))
        plt.show()

    def get_annotations(self, split):
        return self.split_data[split]['annotations']

    def get_img_dir(self, split):
        return self.split_data[split]['img_dir']

    @property
    def interactions(self) -> List:
        return self._interaction_list

    @property
    def predicates(self) -> List:
        return self._predicates

    @property
    def objects(self) -> List:
        return self._objects

    def get_occurrences(self, interaction, split=Splits.TRAIN):
        """
        :param interaction: [tuple or list] <predicate, object> pair
        :param split: [str] one in Splits.TRAIN, Splits.TEST or None (which means across all splits)
        :return: [int] number of occurrences of `interaction` in `split`
        """

        if interaction[0] not in self.predicates or interaction[1] not in self.objects:
            raise ValueError('Invalid interaction (%s)' % str(interaction))

        if split is None:
            splits = [Splits.TRAIN, Splits.TEST]
        else:
            assert split in [Splits.TRAIN, Splits.TEST]
            splits = [split]

        occurrences = 0
        for s in splits:
            for i, inter in enumerate(self.interactions):
                if inter['pred'] == interaction[0] and inter['obj'] == interaction[1]:
                    occurrences += self.split_data[s]['inter_occurrences'][i]
        return occurrences

    def sanity_check(self):
        train_dir = self.split_data[Splits.TRAIN]['img_dir']
        test_dir = self.split_data[Splits.TEST]['img_dir']
        assert len(self.interactions) == 600
        assert len(self.wn_predicate_dict) == 119
        assert len(self.predicate_dict) == 117
        assert len([name for name in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, name))]) == 38118
        assert len([name for name in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, name))]) == 9658
        assert all([y['ing'].endswith('ing') or 'ing_' in y['ing'] or x == 'no_interaction' for x, y in self.predicate_dict.items()])

        for ann in self.split_data[Splits.TRAIN]['annotations']:
            im_w, im_h = ann['img_size'][:2]
            for inter in ann['interactions']:
                if not inter['invis']:
                    for k in ['hum_bbox', 'obj_bbox']:
                        for bbox in inter[k]:
                            assert 0 <= bbox[0] < bbox[2] and \
                                   0 <= bbox[1] < bbox[3] and \
                                   bbox[2] < im_w and \
                                   bbox[3] < im_h, (ann['file'], bbox, im_w, im_h)

        pass
        # # Trying to understand what `count` is
        # inds, counts = np.unique([iid for ann in self.test_annotations for iid in np.unique([inter['id'] for inter in ann['interactions']])],
        #                          return_counts=True)
        # assert np.all(inds == np.arange(600)), inds
        # for i, inter in enumerate(self.interaction_list):
        #     print('%3d %20s %-20s: %4d %4d %4d' % (i, self.predicate_dict[inter['pred_wid']]['name'], inter['obj'],
        #                                            self.predicate_dict[inter['pred_wid']].get('count', -1), counts[i],
        #                                            sum([inter['pred_wid'] == inter2['pred_wid'] for inter2 in self.interaction_list])))

        # TODO print some bounding boxes, maybe?

    def compute_stats(self):
        for split, split_data in self.split_data.items():
            inds, split_counts = np.unique([inter['id'] for ann in split_data['annotations'] for inter in ann['interactions']], return_counts=True)
            assert np.all(inds == np.arange(600)), inds
            split_data['inter_occurrences'] = split_counts

            print('#' * 50, split)
            img_sizes = sorted(set([ann['img_size'][2] for ann in split_data['annotations']]))
            for isize in img_sizes:
                print(isize)

    def load_annotations(self, use_hico_det=True):
        """
        :param use_hico_det: whether you are using the original HICO dataset [False] or HICO-DET [True, default]
        :return: see __init__ for HICO-DET, below for HICO
        """

        def _parse_split(_split):
            # The many "-1"s are due to original values being suited for MATLAB.
            if use_hico_det:
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
            else:
                assert src_anns['anno_%s' % _split].shape[1] == src_anns['list_%s' % _split].size
                _annotations = [{'file': src_anns['list_%s' % _split][i],
                                 'interactions': src_anns['anno_%s' % _split][:, i]}
                                for i in range(src_anns['list_%s' % _split].size)]
            return _annotations

        try:
            with open(self.path_pickle_annotation_file, 'rb') as f:
                d = pickle.load(f)
                train_annotations = d[Splits.TRAIN]
                test_annotations = d[Splits.TEST]
                interaction_list = d['interaction_list']
                wn_pred_dict = d['wn_pred_dict']
                pred_dict = d['pred_dict']
        except FileNotFoundError:
            if use_hico_det:
                src_anns = loadmat(os.path.join(self.data_dir, 'anno_bbox.mat'), squeeze_me=True)
            else:
                # 'anno_train': 600 x 38118 matrix. Associates to each training set images action labels. Specifically, cell (i,j) can contain one
                #       of the four values -1, 0, 1 or NaN according to whether action i is a hard negative, a soft negative/positive,
                #       a hard positive or unknown in image j.
                # 'anno_test': 600 x 9658 matrix. Same format for the training set one.
                # 'list_train' and 'list_set' are respectively 38118- and 9658- dimensional vectors of file names.
                # The parse method merges them into a list of dictionary entries, each of which contains two keys: 'file' and an 'interactions'.
                src_anns = loadmat(os.path.join(self.data_dir, 'anno.mat'), squeeze_me=True)

            train_annotations = _parse_split(_split=Splits.TRAIN)
            test_annotations = _parse_split(_split=Splits.TEST)

            interaction_list, wn_pred_dict, pred_dict = self.parse_interaction_list(src_anns['list_action'])

            with open(self.path_pickle_annotation_file, 'wb') as f:
                pickle.dump({Splits.TRAIN: train_annotations,
                             Splits.TEST: test_annotations,
                             'interaction_list': interaction_list,
                             'wn_pred_dict': wn_pred_dict,
                             'pred_dict': pred_dict,
                             }, f)

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


def main():
    hd = HicoDet()

    # Save lists
    path_objects_file = os.path.join(hd.data_dir, 'objects.txt')
    path_action_synsets_file = os.path.join(hd.data_dir, 'action_synsets.txt')
    with open(path_objects_file, 'w') as f:
        f.write('\n'.join(hd.objects))
    with open(path_action_synsets_file, 'w') as f:
        f.write('\n'.join([' '.join([syn for syn in predicate.get('syn', ['no_interaction'])]) for predicate in hd.wn_predicate_dict.values()]))


if __name__ == '__main__':
    main()
