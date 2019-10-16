import json
import os
from typing import List

import numpy as np

from lib.dataset.hoi_dataset import HoiDataset


class HCVRD:
    def __init__(self):
        self.data_dir = os.path.join('data', 'HCVRD')
        self.objects = self.predicates = self.object_index = self.predicate_index = self.null_object = self.null_predicate = None
        self.triplets = None
        self._load()

    @property
    def human_classes(self) -> List[int]:
        return [self.object_index[o] for o in ['person', 'man', 'woman', 'boy', 'girl', 'child', 'kid', 'baby',
                                               'audience', 'catcher', 'carrier', 'classroom', 'couple', 'cowboy', 'crowd', 'driver', 'friend',
                                               'guard', 'little girl', 'player', 'rider', 'skateboarder', 'skater', 'skier', 'small child',
                                               'snowboarder', 'surfer', 'tennis player', ]]

    @property
    def triplet_str(self):
        return [(self.objects[s], self.predicates[p], self.objects[o]) for s, p, o in self.triplets]

    def _load(self):
        with open(os.path.join(self.data_dir, 'final_data.json'), 'r') as f:
            d = json.load(f)  # {'im_id': [{'predicate', 'object', 'subject', 'obj_box', 'sub_box'}]}
        triplets_str = [[reldata['subject'], reldata['predicate'].strip(), reldata['object']] for imdata in d.values() for reldata in imdata]

        self.objects = sorted({t[i] for t in triplets_str for i in [0, 2]})
        self.object_index = {x: i for i, x in enumerate(self.objects)}
        self.predicates = sorted({t[1] for t in triplets_str})
        self.predicate_index = {x: i for i, x in enumerate(self.predicates)}

        self.triplets = np.array([[self.object_index[s], self.predicate_index[p], self.object_index[o]] for s, p, o in triplets_str])

    def match(self, hoi_ds: HoiDataset, remove_unmatched=False):
        hcvrd_obj_index = {}
        for orig in self.objects:
            obj = orig
            if obj == "ski's":
                obj = 'skis'
            if obj == "hairdryer":
                obj = 'dryer'
            hcvrd_obj_index[obj] = self.object_index[orig]

        hico_to_hcvrd_obj_match = {}
        for orig in hoi_ds.objects:
            obj = orig.replace('_', ' ')
            hico_to_hcvrd_obj_match[orig] = hcvrd_obj_index.get(obj, hcvrd_obj_index.get(obj.split()[-1], None))

        hcvrd_pred_index = {}
        for orig in self.predicates:
            p = orig.split()[0]
            if p == 'lying':
                p = 'lie'
            elif p in ['riding', 'using', 'driving', 'making', 'moving', 'operating', 'racing', 'serving', 'sliding', 'waving']:
                p = p[:-3] + 'e'
            elif p in ['controlling', 'sitting', 'cutting', 'dribbling', 'flipping', 'hugging', 'petting', 'running', 'stirring']:
                p = p[:-4]
            elif p.endswith('ing'):
                p = p[:-3]
            elif p in ['stopped']:
                p = p[:-3]
            elif p in ['says', 'wears', 'chases', 'ties', 'types']:
                p = p[:-1]
            p = ' '.join([p] + orig.split()[1:])
            hcvrd_pred_index[p] = self.predicate_index[orig]

        hico_to_hcvrd_pred_match = {}
        for orig in hoi_ds.actions[1:]:
            hico_to_hcvrd_pred_match[orig] = hcvrd_pred_index.get(orig.replace('_', ' '), None)

        if remove_unmatched:
            hico_to_hcvrd_pred_match = {k: v for k, v in hico_to_hcvrd_pred_match.items() if v is not None}
            hico_to_hcvrd_obj_match = {k: v for k, v in hico_to_hcvrd_obj_match.items() if v is not None}
        return hico_to_hcvrd_pred_match, hico_to_hcvrd_obj_match

    def get_hoi_freq(self, hoi_ds: HoiDataset):
        hico_to_hcvrd_pred_match, hico_to_hcvrd_obj_match = self.match(hoi_ds)

        hcvrd_to_hico_obj = {v: hoi_ds.object_index[k] for k, v in hico_to_hcvrd_obj_match.items() if v is not None}
        hcvrd_to_hico_pred = {v: hoi_ds.action_index[k] for k, v in hico_to_hcvrd_pred_match.items() if v is not None}
        human_classes = set(self.human_classes)

        op_mat = np.zeros((len(hoi_ds.objects), len(hoi_ds.actions)))
        for s, p, o in self.triplets:
            if s in human_classes and p in hcvrd_to_hico_pred.keys() and o in hcvrd_to_hico_obj.keys():
                op_mat[hcvrd_to_hico_obj[o], hcvrd_to_hico_pred[p]] += 1

        return op_mat


def main():
    from lib.dataset.hico import Hico
    hcvrd = HCVRD()
    hico = Hico()

    print(hcvrd.human_classes)
    print('\n'.join(hcvrd.objects))

    hico_to_hcvrd_pred_match, hico_to_hcvrd_obj_match = hcvrd.match(hico)
    mp, mo = sorted([k for k, v in hico_to_hcvrd_pred_match.items() if v is None]), sorted([k for k, v in hico_to_hcvrd_obj_match.items() if v is None])
    print(mp)
    print(mo)


if __name__ == '__main__':
    main()
