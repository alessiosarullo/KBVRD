import sys

import h5py
import numpy as np

from config import cfg
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.dataset.utils import Splits, Minibatch


def save_feats():
    batch_size = 1
    flipping = False

    sys.argv += ['--model', 'base', '--save_dir', 'fake']  # fake required arguments
    sys.argv += ['--batch_size', str(batch_size), '--val_ratio', '0']
    cfg.parse_args()

    train_split = HicoDetInstanceSplit.get_split(split=Splits.TRAIN, flipping_prob=float(flipping))
    test_split = HicoDetInstanceSplit.get_split(split=Splits.TEST, flipping_prob=float(flipping))
    assert Splits.VAL not in HicoDetInstanceSplit._splits

    for split, hds in [(Splits.TRAIN, train_split),
                       (Splits.TEST, test_split),
                       ]:
        hd_loader = hds.get_loader(batch_size=batch_size, shuffle=False, drop_last=False)

        precomputed_feats_fn = cfg.program.precomputed_feats_file_format % (cfg.model.rcnn_arch, split.value)
        feat_file = h5py.File(precomputed_feats_fn, 'r+')

        try:
            if 'img_infos' in feat_file:
                del feat_file['img_infos']

            all_img_infos = []
            for im_i, im_data in enumerate(hd_loader):
                im_data = im_data  # type: Minibatch
                assert len(im_data.other_ex_data) == 1
                assert im_data.other_ex_data[0]['flipped'] == flipping

                im_infos = np.array([*im_data.other_ex_data[0]['im_size'], im_data.other_ex_data[0]['im_scale']])
                all_img_infos.append(im_infos)
                if im_i % 10 == 0 or im_i == len(hd_loader) - 1:
                    print('Image %6d/%d' % (im_i, len(hd_loader)))

            assert len(all_img_infos) == len(hd_loader)
            all_img_infos = np.stack([x for x in all_img_infos if x is not None], axis=0)
            assert all_img_infos.shape[0] == len(hd_loader)
            feat_file.create_dataset('img_infos', data=all_img_infos)

        finally:
            feat_file.close()
            print('%s feat file closed.' % split.value)


if __name__ == '__main__':
    save_feats()
