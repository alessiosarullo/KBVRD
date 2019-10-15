import json
import os
from typing import List

import h5py
import numpy as np

from lib.dataset.hicodet.hicodet import HicoDet


class VGSGG:
    def __init__(self):
        self.data_dir = os.path.join('data', 'VG-SGG')
        self.objects = self.predicates = self.object_index = self.predicate_index = self.null_object = self.null_predicate = None
        self.triplets = None
        self._load()

    @property
    def human_classes(self) -> List[int]:
        return [self.object_index[o] for o in ['person', 'people', 'guy', 'man', 'men', 'player', 'skier', 'lady', 'woman', 'boy', 'girl', 'child',
                                               'kid']]

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

    def match(self, hd: HicoDet, remove_unmatched=False):
        hd_to_vgsgg_obj_match = {}
        for orig in hd.objects:
            obj = orig.replace('_', ' ')
            hd_to_vgsgg_obj_match[orig] = self.object_index.get(obj, self.object_index.get(obj.split()[-1], None))

        vgsgg_pred_index = {self.null_predicate: self.predicate_index[self.null_predicate]}
        for orig in self.predicates[1:]:
            s = orig.split()
            p = s[0]
            if p == 'riding':
                p = 'ride'
            elif p == 'sitting':
                p = 'sit'
            elif p == 'using':
                p = 'use'
            elif p == 'lying':
                p = 'lie'
            elif p.endswith('ing'):
                p = p[:-3]
            elif p == 'says':
                p = 'say'
            elif p == 'wears':
                p = 'wear'

            if len(s) == 2 and s[1] in {'on', 'at', 'under', 'with'}:
                p = f'{p} {s[1]}'
            vgsgg_pred_index[p] = self.predicate_index[orig]

        hd_to_vgsgg_pred_match = {}
        for orig in hd.actions[1:]:
            # s = orig.split('_')
            # if '_' in orig and s[1] == 'on':
            #     p = s[0]
            # else:
            #     p = orig
            hd_to_vgsgg_pred_match[orig] = vgsgg_pred_index.get(orig.replace('_', ' '), None)

        if remove_unmatched:
            hd_to_vgsgg_pred_match = {k: v for k, v in hd_to_vgsgg_pred_match.items() if v is not None}
            hd_to_vgsgg_obj_match = {k: v for k, v in hd_to_vgsgg_obj_match.items() if v is not None}

        return hd_to_vgsgg_pred_match, hd_to_vgsgg_obj_match

    def get_hoi_freq(self, hd: HicoDet):
        hd_to_vgsgg_pred_match, hd_to_vgsgg_obj_match = self.match(hd)

        vgsgg_to_hd_obj = {v: hd.object_index[k] for k, v in hd_to_vgsgg_obj_match.items() if v is not None}
        vgsgg_to_hd_pred = {v: hd.action_index[k] for k, v in hd_to_vgsgg_pred_match.items() if v is not None}
        human_classes = set(self.human_classes)

        op_mat = np.zeros((len(hd.objects), len(hd.actions)))
        for s, p, o in self.triplets:
            if s in human_classes and p in vgsgg_to_hd_pred.keys() and o in vgsgg_to_hd_obj.keys():
                op_mat[vgsgg_to_hd_obj[o], vgsgg_to_hd_pred[p]] += 1

        return op_mat

    #
    # def match_preds():
    #     vg = VGSGG()
    #     hd = HicoDet()
    #
    #     vgp = {vg.null_predicate: vg.null_predicate}
    #     for orig in vg.predicates[1:]:
    #         p = orig.split()[0]
    #         if p == 'riding':
    #             p = 'ride'
    #         elif p == 'sitting':
    #             p = 'sit'
    #         elif p == 'using':
    #             p = 'use'
    #         elif p.endswith('ing'):
    #             p = p[:-3]
    #         elif p == 'says':
    #             p = 'say'
    #         elif p == 'wears':
    #             p = 'wear'
    #         vgp[p] = orig
    #
    #     hdp = {hd.null_interaction: hd.null_interaction}
    #     for orig in hd.actions[1:]:
    #         p = orig.split('_')[0]
    #         hdp[p] = orig
    #
    #     svgp = set(vgp.keys())
    #     shdp = set(hdp.keys())
    #     return sorted([vgp[x] for x in svgp & shdp]), sorted([vgp[x] for x in svgp - shdp]), sorted([hdp[x] for x in shdp - svgp])
    #
    #
    # def match_objs():
    #     vg = VGSGG()
    #     hd = HicoDet()
    #
    #     vgp = {vg.null_object: vg.null_object}
    #     for orig in vg.objects[1:]:
    #         vgp[orig] = orig
    #
    #     hdp = {}
    #     for orig in hd.objects:
    #         p = orig.split('_')[-1]
    #         hdp[p] = orig
    #
    #     svgp = set(vgp.keys())
    #     shdp = set(hdp.keys())
    #     return sorted([hdp[x] for x in svgp & shdp]), sorted([vgp[x] for x in svgp - shdp]), sorted([hdp[x] for x in shdp - svgp])


def main():
    vg = VGSGG()
    hd = HicoDet()

    hd_to_vgsgg_pred_match, hd_to_vgsgg_obj_match = vg.match(hd)
    for k, v in hd_to_vgsgg_pred_match.items():
        if v is not None:
            print(k, v)

    print()

    for k, v in hd_to_vgsgg_obj_match.items():
        if v is not None:
            print(k, v)


if __name__ == '__main__':
    main()
