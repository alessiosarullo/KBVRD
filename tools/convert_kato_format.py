import os
import pickle

import numpy as np

from lib.dataset.hico.hico import Hico
from lib.dataset.hico.hico_split import HicoSplit, Splits

from config import cfg


def main(dataset='HICO'):
    if dataset == 'HICO':
        kato_dir = os.path.join(cfg.data_root, dataset, 'Kato')

        hico = Hico()

        # Check that indices are consistent
        with open(os.path.join(kato_dir, 'hico_list_hoi.csv'), 'r') as f:
            interactions_classes = [l.strip().split(',') for l in f.readlines() if l.strip()]
        for idx, obj, act in interactions_classes:
            idx = int(idx)
            # print(f'{idx:3d} {act:20s} {obj:15s}')
            hico_act_idx, hico_obj_idx = hico.interactions[idx, :]
            assert hico.objects[hico_obj_idx] == obj or (hico.objects[hico_obj_idx] == 'hair_dryer' and obj == 'hair_drier')
            assert hico.actions[hico_act_idx] == act or (act == hico.null_interaction.strip('_'))

        # Load seen interaction indices
        data = {}
        for split in ['1A', '1B', '2A', '2B']:
            with open(os.path.join(kato_dir, f'hico_{split}.txt'), 'r') as f:
                interaction_inds = np.array([int(l.strip()) for l in f.readlines()])

            interactions = hico.interactions[interaction_inds]
            actions_inds = np.array(sorted(set(interactions[:, 0].tolist())))
            objects_inds = np.array(sorted(set(interactions[:, 1].tolist())))
            print(split)
            print(len(interaction_inds), interaction_inds)
            print(len(actions_inds), actions_inds)
            print(len(objects_inds), objects_inds)
            print()
            data[split] = {'i': interaction_inds, 'a': actions_inds, 'o': objects_inds}

        # Prints and checks
        obj_1 = set(data['1A']['o'].tolist()) | set(data['1B']['o'].tolist())
        obj_2 = set(data['2A']['o'].tolist()) | set(data['2B']['o'].tolist())
        print(len(obj_1), len(obj_2))
        # assert len(obj_1) == 80 // 2 and len(obj_2) == 80 // 2  # This is not true
        act_1 = set(data['1A']['a'].tolist()) | set(data['2A']['a'].tolist())
        act_2 = set(data['1B']['a'].tolist()) | set(data['2B']['a'].tolist())
        print(len(act_1), len(act_2))
        assert len(act_1) == 117 // 2 or len(act_1) == 117 // 2 + 1
        assert len(act_2) == 117 // 2 or len(act_2) == 117 // 2 + 1
        print(sum([len(data[split]['i']) for split in ['1A', '1B', '2A', '2B']]))

        # Write zero-shot file
        seen_obj = data['1A']['o']
        seen_act = np.array(sorted(set(data['1A']['a'].tolist()) | {0}))
        with open('zero-shot_inds/seen_inds_2.pkl.push', 'wb') as f:
            pickle.dump({'train': {'obj': seen_obj, 'pred': seen_act}}, f)

        # Check image data
        def load_kato_ann(fn, img_inv_ind):
            with open(os.path.join(kato_dir, fn), 'r') as f:
                kato_ann_str = [l.strip().split(',') for l in f.readlines() if l.strip()]
                kato_ann = np.zeros_like(hico.split_annotations[split])
                for l in kato_ann_str:
                    fn = l[0]
                    img_interactions = [int(i) for i in l[1].split(';')]
                    # print(fn, img_interactions)
                    kato_ann[img_inv_ind[fn], np.array(img_interactions)] = 1
            return kato_ann

        split = Splits.TRAIN
        img_inv_ind = {fn: i for i, fn in enumerate(hico.split_filenames[split])}
        kato_ann = load_kato_ann('hico_train_1A.csv', img_inv_ind)
        hs = HicoSplit(split, hico, object_inds=seen_obj, action_inds=seen_act[1:])  # exclude null fom comparison
        labels = (hs.labels > 0)  # -1s are converted to 0s
        assert np.all(labels == kato_ann)

        split = Splits.TEST
        img_inv_ind = {fn: i for i, fn in enumerate(hico.split_filenames[split])}
        kato_ann = load_kato_ann('hico_test.csv', img_inv_ind)
        hs = HicoSplit(split, hico)
        labels = (hs.labels > 0)  # -1s are converted to 0s
        assert np.all(labels == kato_ann)

    elif dataset == 'VG':
        pass  # TODO
    else:
        raise ValueError('Unknown dataset.')


if __name__ == '__main__':
    main()
