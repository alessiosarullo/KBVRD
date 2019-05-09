import os
import h5py
import json
from typing import List

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.io import loadmat

from lib.dataset.utils import Splits


class VGSGG:
    def __init__(self):
        self.data_dir = os.path.join('data', 'VG-SGG')
        self.objects = self.predicates = self.object_index = self.predicate_index = self.null_object = self.null_predicate = None
        self.triplets = None
        self._load()

    @property
    def human_class(self) -> int:
        raise NotImplementedError()
        return self._obj_class_index['person']

    @property
    def triplet_str(self):
        return [(self.objects[s], self.predicates[p], self.objects[o]) for s, p, o in self.triplets]

    def _load(self):
        vg_data_file = h5py.File(os.path.join(self.data_dir, 'VG-SGG.h5'), 'r')
        # Keys: ['active_object_mask', 'boxes_1024', 'boxes_512', 'img_to_first_box', 'img_to_first_rel', 'img_to_last_box', 'img_to_last_rel',
        #        'labels', 'predicates', 'relationships', 'split']

        rel = vg_data_file['relationships'][:]
        l = vg_data_file['labels'][:].squeeze(axis=1)
        p = vg_data_file['predicates'][:].squeeze(axis=1)
        self.triplets = np.stack([l[rel[:, 0]], p, l[rel[:, 1]]], axis=1)

        with open(os.path.join(self.data_dir, 'VG-SGG-dicts.json'), 'r') as f:
            dicts = json.load(f)  # Keys: ['object_count', 'idx_to_label', 'predicate_to_idx', 'predicate_count', 'idx_to_predicate', 'label_to_idx']

        objects = dicts['idx_to_label']
        self.objects = ['__no_object__'] + [objects[str(i + 1)] for i in range(len(objects))]
        self.object_index = {x: i for i, x in enumerate(self.objects)}
        self.null_object = self.objects[0]
        assert all([v == self.object_index[k] for k, v in dicts['label_to_idx'].items()]) and \
               set(dicts['label_to_idx'].keys()) == (set(self.object_index.keys()) - {self.null_object})

        predicates = dicts['idx_to_predicate']
        self.predicates = ['__no_predicate__'] + [predicates[str(i + 1)] for i in range(len(predicates))]
        self.predicate_index = {x: i for i, x in enumerate(self.predicates)}
        self.null_predicate = self.predicates[0]
        assert all([v == self.predicate_index[k] for k, v in dicts['predicate_to_idx'].items()]) and \
               set(dicts['predicate_to_idx'].keys()) == (set(self.predicate_index.keys()) - {self.null_predicate})


def match():
    from lib.dataset.hicodet_driver import HicoDet
    vg = VGSGG()
    hd = HicoDet()

    vgp = {vg.null_predicate: vg.null_predicate}
    for orig in vg.predicates[1:]:
        p = orig.split()[0]
        if p == 'riding':
            p = 'ride'
        elif p == 'sitting':
            p = 'sit'
        elif p == 'using':
            p = 'use'
        elif p.endswith('ing'):
            p = p[:-3]
        elif p == 'says':
            p = 'say'
        elif p == 'wears':
            p = 'wear'
        vgp[p] = orig

    hdp = {hd.null_interaction: hd.null_interaction}
    for orig in hd.predicates[1:]:
        p = orig.split('_')[0]
        hdp[p] = orig

    svgp = set(vgp.keys())
    shdp = set(hdp.keys())
    return sorted([vgp[x] for x in svgp & shdp]), sorted([vgp[x] for x in svgp - shdp]), sorted([hdp[x] for x in shdp - svgp])


def main():
    from lib.dataset.hicodet_driver import HicoDet
    vg = VGSGG()
    hd = HicoDet()

    m, v, h = match()
    print(m)
    print(v)
    print(h)


if __name__ == '__main__':
    main()
