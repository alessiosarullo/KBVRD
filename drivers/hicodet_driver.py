import os
import pickle

import numpy as np
from scipy.io import loadmat


class HicoDetLoader:
    def __init__(self):
        """
        Class attributes:
            - interaction_list: [list] The 600 interactions in HICO-DET. Each element consists of:
                - 'obj': [str] The name of the object of the action (i.e., the target).
                - 'predicate_wid': [str] The wordnet ID of the action.
            - predicate_dict: [dict] The 120 possible actions, including a null one (called 'no_interaction'). Keys are wordnets IDs and the
                null action is assigned a fake ID 'v0'. Each element except the null action contains:
                    - 'wid': [str] The wordnet ID (it's the same as the key).
                    - 'wname': [str] The name of the wordnet entry this actions refers to. It is in the form VERB.v.NUM, where VERB is the verb
                        describing the action and NUM is an index used to disambiguate between homonyms.
                    - 'name': [str] The first part of 'wname' (i.e., VERB in the example above).
                    - 'id': [int] A number I have not understood the use of.
                    - 'count': [int] Another number I have not understood the use of.
                    - 'syn': [list] Set of synonyms
                    - 'def': [str] A definition
                    - 'ex': [str] An example (sometimes not provided)
                    - base forms: [list] Base forms the action is found in in the dataset. It usually consists of just one element that is the
                        same as 'name', but there are two exceptions:
                          1) sometimes actions come in different forms, e.g., instances of 'sit' are 'sit_on' and 'sit_at'
                          2) sometimes the base form is different from the name, e.g., the action 'ignite' is only found in the dataset under the
                            base form 'light'
                    - ing forms: [list] -ing forms of the corresponding base forms.
                The null actions only contains 'wid', 'name', 'base forms', 'ing forms.
            - train_annotations, test_annotations: [list] Each entry is a dictionary with the following items:
                - 'file': [str] The file name
                - 'img_size': [array] Image size expressed in [width, height, depth]
                - 'interactions': [list] Each entry has:
                        - 'id': [int] The id of the interaction in `interaction_list`.
                        - 'invis': [bool] Whether the interaction is invisible or not. It does NOT necesserily mean that it is not in the image.
                    If 'invis' is False then there are three more fields:
                        - 'hum_bbox': [array] Hx4 matrix of (x1, y1, x2, y2) coordinates for each bounding box belonging to a human.
                        - 'obj_bbox': [array] Ox4 matrix of (x1, y1, x2, y2) coordinates for each bounding box belonging to an object.
                        - 'conn': [array] Cx2 with a pair of human-object indices for each interaction
        """
        # FIXME what are the 'id' and 'count' field for?

        self.data_dir = os.path.join('data', 'HICO-DET')
        self.train_img_dir = os.path.join(self.data_dir, 'images', 'train2015')
        self.test_img_dir = os.path.join(self.data_dir, 'images', 'test2015')
        self.path_pickle_annotation_file = os.path.join(self.data_dir, 'annotations.pkl')
        self.path_objects_file = os.path.join(self.data_dir, 'objects.txt')
        self.path_action_synsets_file = os.path.join(self.data_dir, 'action_synsets.txt')

        self.train_annotations, self.test_annotations, self.interaction_list, self.predicate_dict = self.load_annotations(use_hico_det=True)

        # Derived structures
        self._int_names_list = [(self.predicate_dict[inter['predicate_wid']]['name'], inter['obj']) for inter in self.interaction_list]

        # Statistics
        self._num_interaction_occurrences = None
        self.compute_stats()

        # Sanity check
        self.sanity_check()

    @property
    def predicates(self):
        return [v['name'] for v in self.predicate_dict.values()]

    @property
    def objects(self):
        return sorted(set([inter['obj'] for inter in self.interaction_list]))

    def get_occurrences(self, interaction, split='train'):
        """
        :param interaction: [tuple or list] <predicate, object> pair
        :param split: [str] one in 'train', 'test' or 'both'
        :return: [int] number of occurrences of `interaction` in `split`
        """

        if interaction[0] not in self.predicates or interaction[1] not in self.objects:
            raise ValueError('Invalid interaction (%s)' % str(interaction))

        if split is None:
            split = 'both'
        assert split in ['train', 'test', 'both']

        for iid, inter in enumerate(self._int_names_list):
            if tuple(interaction) == inter:
                interaction_id = iid
                break
        else:
            return 0

        occurrences = 0
        if split == 'train' or split == 'both':
            occurrences += self._num_interaction_occurrences['train'][interaction_id]
        if split == 'test' or split == 'both':
            occurrences += self._num_interaction_occurrences['test'][interaction_id]
        return occurrences

    def compute_stats(self):
        inds, train_counts = np.unique([inter['id'] for ann in self.test_annotations for inter in ann['interactions']], return_counts=True)
        assert np.all(inds == np.arange(600)), inds
        inds, test_counts = np.unique([inter['id'] for ann in self.test_annotations for inter in ann['interactions']], return_counts=True)
        assert np.all(inds == np.arange(600)), inds
        self._num_interaction_occurrences = {'train': train_counts,
                                             'test': test_counts}

    def save_lists(self):
        with open(self.path_objects_file, 'w') as f:
            f.write('\n'.join(self.objects))
        with open(self.path_action_synsets_file, 'w') as f:
            f.write('\n'.join([' '.join([syn for syn in predicate.get('syn', ['no_interaction'])]) for predicate in self.predicate_dict.values()]))

    def load_annotations(self, use_hico_det=True):
        """
        :param use_hico_det: whether you are using the original HICO dataset [False] or HICO-DET [True, default]
        :return: see __init__ for HICO-DET, below for HICO
        """

        def _parse_split(_split):
            if use_hico_det:
                _annotations = []
                for _src_ann in src_anns['bbox_%s' % _split]:
                    _ann = {'file': _src_ann[0],
                            'img_size': np.array([int(_src_ann[1][field]) for field in ['width', 'height', 'depth']], dtype=np.int),
                            'interactions': []}
                    for _inter in np.atleast_1d(_src_ann[2]):
                        _new_inter = {
                            'id': int(_inter['id']) - 1,
                            'invis': bool(_inter['invis']),
                        }
                        if not _new_inter['invis']:
                            _new_inter['hum_bbox'] = np.atleast_2d(np.array([_inter['bboxhuman'][c] for c in _inter['bboxhuman'].dtype.fields],
                                                                            dtype=np.int).T)
                            _new_inter['obj_bbox'] = np.atleast_2d(np.array([_inter['bboxobject'][c] for c in _inter['bboxobject'].dtype.fields],
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
                train_annotations = d['train']
                test_annotations = d['test']
                interaction_list = d['interaction_list']
                predicate_dict = d['predicate_dict']
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

            train_annotations = _parse_split(_split='train')
            test_annotations = _parse_split(_split='test')

            interaction_list, predicate_dict = self.parse_interaction_list(src_anns['list_action'])

            with open(self.path_pickle_annotation_file, 'wb') as f:
                pickle.dump({'train': train_annotations,
                             'test': test_annotations,
                             'interaction_list': interaction_list,
                             'predicate_dict': predicate_dict,
                             }, f)

        return train_annotations, test_annotations, interaction_list, predicate_dict

    def sanity_check(self):
        assert len([name for name in os.listdir(self.train_img_dir) if os.path.isfile(os.path.join(self.train_img_dir, name))]) == 38118
        assert len([name for name in os.listdir(self.test_img_dir) if os.path.isfile(os.path.join(self.test_img_dir, name))]) == 9658
        assert all([x.endswith('ing') or 'ing_' in x or x == self.predicate_dict['v0']['name']
                    for y in self.predicate_dict.values() for x in y['ing forms']])

        # # Trying to understand what `count` is
        # inds, counts = np.unique([iid for ann in self.test_annotations for iid in np.unique([inter['id'] for inter in ann['interactions']])],
        #                          return_counts=True)
        # assert np.all(inds == np.arange(600)), inds
        # for i, inter in enumerate(self.interaction_list):
        #     print('%3d %20s %-20s: %4d %4d %4d' % (i, self.predicate_dict[inter['predicate_wid']]['name'], inter['obj'],
        #                                            self.predicate_dict[inter['predicate_wid']].get('count', -1), counts[i],
        #                                            sum([inter['predicate_wid'] == inter2['predicate_wid'] for inter2 in self.interaction_list])))

        # TODO print some bounding boxes, maybe?

    @staticmethod
    def parse_interaction_list(src_interaction_list):
        predicate_dict = {}
        interaction_list = []

        for i, interaction_ann in enumerate(src_interaction_list):
            fields = interaction_ann[-2].dtype.fields
            if fields is None:  # Null interaction
                for j, s in enumerate(interaction_ann):
                    if j < 3:
                        if j > 0:
                            assert s == 'no_interaction'
                        assert isinstance(s, str)
                    else:
                        assert s.size == 0
                action_ann = {'wid': 'v0', 'name': interaction_ann[1]}
            else:
                action_ann = {}
                for f in fields:
                    fvalue = str(interaction_ann[-2][f])
                    try:
                        fvalue = int(fvalue)
                    except ValueError:
                        pass

                    if f == 'name':
                        action_ann['wname'] = fvalue
                    elif f == 'syn':
                        action_ann[f] = list(set(fvalue.split(' ')))
                    elif f == 'ex':
                        action_ann[f] = fvalue if fvalue != '[]' else ''
                    else:
                        action_ann[f] = fvalue
                assert 'name' not in action_ann
                action_ann['name'] = action_ann['wname'].split('.')[0]

            # Add to the interaction list
            action_id = action_ann['wid']
            new_action_ann = {'obj': interaction_ann[0], 'predicate_wid': action_id}
            interaction_list.append(new_action_ann)

            # Add to the action dictionary
            d_action_ann = predicate_dict.setdefault(action_id, action_ann)
            if interaction_ann[1] not in d_action_ann.get('base forms', []):
                d_action_ann['base forms'] = d_action_ann.get('base forms', []) + [interaction_ann[1]]
            if interaction_ann[2] not in d_action_ann.get('ing forms', []):
                d_action_ann['ing forms'] = d_action_ann.get('ing forms', []) + [interaction_ann[2]]
            assert {k: v for k, v in d_action_ann.items() if 'forms' not in k} == {k: v for k, v in action_ann.items() if 'forms' not in k}, \
                '\n%s\n%s' % (d_action_ann, action_ann)

        # Sort
        sorted_wids = sorted(list(predicate_dict.keys()))
        predicate_dict = {k: predicate_dict[k] for k in sorted_wids}

        return interaction_list, predicate_dict


def main():
    data_loader = HicoDetLoader()
    data_loader.save_lists()


if __name__ == '__main__':
    main()
