import os
import h5py
import json
from typing import List

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.io import loadmat

from lib.dataset.utils import Splits


class HCVRD:
    def __init__(self):
        self.data_dir = os.path.join('data', 'HCVRD')
        self.objects = self.predicates = self.object_index = self.predicate_index = self.null_object = self.null_predicate = None
        self.triplets = None
        self._load()

    @property
    def human_class(self) -> int:
        raise NotImplementedError()

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

        # Just to check
        with open(os.path.join(self.data_dir, 'predicates.txt'), 'r') as f:
            predicates = sorted(set([l.strip() for l in f.readlines()]))
        assert predicates == self.predicates

        self.triplets = np.array([[self.object_index[s], self.predicate_index[p], self.object_index[o]] for s, p, o in triplets_str])


def match():
    from lib.dataset.hicodet_driver import HicoDet
    hcvrd = HCVRD()
    hd = HicoDet()

    hcvrd_preds = set()
    for orig in hcvrd.predicates[1:]:
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
        hcvrd_preds.add(' '.join([p] + orig.split()[1:]))

    hd_preds = set()
    for orig in hd.predicates[1:]:
        p = orig.split('_')[0]
        hd_preds.add(' '.join([p] + orig.split('_')[1:]))

    return sorted(hd_preds - hcvrd_preds)


def main():
    from lib.dataset.hicodet_driver import HicoDet
    vg = HCVRD()
    hd = HicoDet()

    m = match()
    print(m)


if __name__ == '__main__':
    main()
