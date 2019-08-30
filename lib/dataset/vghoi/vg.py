import os

import numpy as np

from config import cfg
from lib.dataset.utils import Splits
from lib.dataset.hoi_dataset import HoiDataset


class VGHOI(HoiDataset):
    def __init__(self):
        ds_dir = os.path.join(cfg.data_root, 'VG')
        kato_dir = os.path.join(ds_dir, 'Kato')

        with open(os.path.join(kato_dir, 'VG_list_hoi.csv'), 'r') as f:
            interactions_str = [l.strip().split(',') for l in f.readlines() if l.strip()]
        idxs, inter_objects, inter_actions = [], [], []
        for idx, obj, act in interactions_str:
            idxs.append(int(idx))
            inter_objects.append(obj)
            inter_actions.append(act)
        assert idxs == list(range(len(idxs)))
        object_classes = sorted(set(inter_objects))
        null_action = '__no_interaction__'
        action_classes = [self.null_action] + sorted(set(inter_actions))
        interactions_str = zip(inter_actions, inter_objects)

        def load_kato_data(data_filename):
            with open(os.path.join(kato_dir, data_filename), 'r') as f:
                data_str = [l.strip().split(',') for l in f.readlines() if l.strip()]
            data_str = [[os.path.splitext(l[0])[0], l[1]] for l in data_str]
            filenames = sorted({l[0] for l in data_str})
            fn_inv_index = {fn: i for i, fn in enumerate(filenames)}
            labels = np.zeros([len(filenames), self.num_interactions])
            for l in data_str:
                anns = np.array([int(x) for x in l[1].split(';')])
                labels[fn_inv_index[l[0]], anns] = 1
            return filenames, labels
        filenames_per_split = {}
        annotations_per_split = {}
        filenames_per_split[Splits.TRAIN], annotations_per_split[Splits.TRAIN] = load_kato_data('VG_train_1A2B.csv')
        filenames_per_split[Splits.TEST], annotations_per_split[Splits.TEST] = load_kato_data('VG_test.csv')

        super(VGHOI, self).__init__(object_classes, action_classes, null_action, interactions_str, filenames_per_split, annotations_per_split)

        with open(os.path.join(kato_dir, 'common_class.csv'), 'r') as f:
            self.common_interactions = np.array(sorted([int(l.strip()) for l in f.readlines() if l.strip()]))
        self.img_dir = os.path.join(ds_dir, 'all_VG_100K')

    def get_img_dir(self, split):
        return self.img_dir


if __name__ == '__main__':
    VGHOI()
