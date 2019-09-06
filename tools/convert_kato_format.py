import os
import pickle
import argparse

import numpy as np

from config import cfg
from lib.dataset.hico import Hico, HicoSplit
from lib.dataset.vghoi import VGHoi, VGHoiSplit
from lib.dataset.utils import Splits


# Check image data
def load_kato_ann(file_path, dataset, split, img_inv_ind_per_split):
    with open(file_path, 'r') as f:
        kato_ann_str = [l.strip().split(',') for l in f.readlines() if l.strip()]
        kato_ann = np.zeros_like(dataset.split_annotations[split])
        for l in kato_ann_str:
            fn = l[0]
            img_interactions = [int(i) for i in l[1].split(';')]
            # print(fn, img_interactions)
            kato_ann[img_inv_ind_per_split[split][fn], np.array(img_interactions)] = 1
    return kato_ann


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['hico', 'VG'])
    args = parser.parse_args()
    ds_name = args.dataset

    kato_dir = os.path.join(cfg.data_root, ds_name.upper(), 'Kato')

    if ds_name == 'hico':
        dataset = Hico()
        dataset_split_class = HicoSplit
        zs_output_file = os.path.join('zero-shot_inds', 'seen_inds_0.pkl.push')
        img_inv_ind_per_split = {split: {fn: i for i, fn in enumerate(dataset.split_filenames[split])} for split in [Splits.TRAIN, Splits.TEST]}
    elif ds_name == 'VG':
        dataset = VGHoi()
        dataset_split_class = VGHoiSplit
        zs_output_file = os.path.join('zero-shot_inds', 'seen_inds_vghoi.pkl.push')
        img_inv_ind_per_split = {split: {fn.replace('.jpg', '.npy'): i for i, fn in enumerate(dataset.split_filenames[split])}
                                 for split in [Splits.TRAIN, Splits.TEST]}
    else:
        raise ValueError('Unknown dataset.')

    # Check that indices are consistent
    with open(os.path.join(kato_dir, f'{ds_name}_list_hoi.csv'), 'r') as f:
        interactions_classes = [l.strip().split(',') for l in f.readlines() if l.strip()]
    for idx, obj, act in interactions_classes:
        idx = int(idx)
        # print(f'{idx:3d} {act:20s} {obj:15s}')
        my_act_idx, my_obj_idx = dataset.interactions[idx, :]
        assert dataset.objects[my_obj_idx] == obj or (dataset.objects[my_obj_idx] == 'hair_dryer' and obj == 'hair_drier')
        assert dataset.actions[my_act_idx] == act or (act == dataset.null_action.strip('_'))

    # Load seen interaction indices
    data = {}
    for split in ['1A', '1B', '2A', '2B']:
        with open(os.path.join(kato_dir, f'{ds_name}_{split}.txt'), 'r') as f:
            interaction_inds = np.array([int(l.strip()) for l in f.readlines()])

        interactions = dataset.interactions[interaction_inds]
        actions_inds = np.array(sorted(set(interactions[:, 0].tolist())))
        objects_inds = np.array(sorted(set(interactions[:, 1].tolist())))
        print(split)
        # print(len(interaction_inds), interaction_inds)
        # print(len(actions_inds), actions_inds)
        # print(len(objects_inds), objects_inds)
        print()
        data[split] = {'i': interaction_inds, 'a': actions_inds, 'o': objects_inds}

    # Prints and checks
    obj_1 = set(data['1A']['o'].tolist()) | set(data['1B']['o'].tolist())
    obj_2 = set(data['2A']['o'].tolist()) | set(data['2B']['o'].tolist())
    print([len(data[split]['o']) for split in ['1A', '1B', '2A', '2B']], len(obj_1), len(obj_2), len(obj_1 | obj_2))
    act_1 = set(data['1A']['a'].tolist()) | set(data['2A']['a'].tolist())
    act_2 = set(data['1B']['a'].tolist()) | set(data['2B']['a'].tolist())
    print([len(data[split]['a']) for split in ['1A', '1B', '2A', '2B']], len(act_1), len(act_2), len(act_1 | act_2))
    print(sum([len(data[split]['i']) for split in ['1A', '1B', '2A', '2B']]))

    # Write zero-shot file
    seen_obj = data['1A']['o']
    seen_act = np.array(sorted(set(data['1A']['a'].tolist()) | {0}))
    with open(zs_output_file, 'wb') as f:
        pickle.dump({'train': {'obj': seen_obj, 'act': seen_act}}, f)

    # Check label consistency
    # This doesn't work for VG because there is some overlap of both objects and actions between 1A and 2B
    if dataset == 'hico':
        split = Splits.TRAIN
        kato_ann = load_kato_ann(os.path.join(kato_dir, f'{ds_name}_train_1A.csv'), dataset, split, img_inv_ind_per_split)
        hs = dataset_split_class(split, dataset, object_inds=seen_obj, action_inds=seen_act[1:])  # exclude null fom comparison
        labels = (hs.labels > 0).astype(np.float)  # -1s are converted to 0s
        assert np.all(labels == kato_ann)

        split = Splits.TEST
        kato_ann = load_kato_ann(os.path.join(kato_dir, f'{ds_name}_test.csv'), dataset, split, img_inv_ind_per_split)
        hs = dataset_split_class(split, dataset)
        labels = (hs.labels > 0).astype(np.float)  # -1s are converted to 0s
        assert np.all(labels == kato_ann)


if __name__ == '__main__':
    main()
