import os
import pickle

import numpy as np

from lib.dataset.hico.hico import Hico


def main(dataset='hico'):
    if dataset == 'hico':
        ds_dir = 'HICO'
        kato_dir = os.path.join('zero-shot_inds', 'Kato', ds_dir)

        hico = Hico()

        # Check that indices are consistent
        with open(os.path.join(kato_dir, 'hico_list_hoi.csv'), 'r') as f:
            interactions_str = [l.strip().split(',') for l in f.readlines() if l.strip()]
        for idx, obj, act in interactions_str:
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

        with open('zero-shot_inds/seen_inds_2.pkl.push', 'wb') as f:
            seen_obj = data['1A']['o']
            seen_act = np.array(sorted(set(data['1A']['a'].tolist()) | {0}))
            pickle.dump({'train': {'obj': seen_obj, 'pred': seen_act}}, f)
    elif dataset == 'VG':
        ds_dir = 'Visual_Genome_HOI'
    else:
        raise ValueError('Unknown dataset.')


if __name__ == '__main__':
    main()
