import sys

import h5py
import numpy as np
import torch
import torch.utils.data
from config import cfg
from lib.dataset.hico.hico_split import HicoSplit, ImgEntry
from lib.dataset.utils import Splits
from torchvision.models import resnet152


def save_feats():
    sys.argv += ['--val_ratio', '0']
    cfg.parse_args(fail_if_missing=False)

    if cfg.program.debug:
        try:  # PyCharm debugging
            print('Starting remote debugging (resume from debug server)')
            import pydevd_pycharm
            pydevd_pycharm.settrace('130.88.195.105', port=16004, stdoutToServer=True, stderrToServer=True)
            print('Remote debugging activated.')
        except:
            print('Remote debugging failed.')
            raise

    splits = HicoSplit.get_splits()
    assert Splits.VAL not in splits

    vm = resnet152(pretrained=True)
    if torch.cuda.is_available():
        vm.cuda()
    else:
        print('!!!!!!!!!!!!!!!!! Running on CPU!')
    vm.eval()
    for split in [Splits.TRAIN, Splits.TEST]:
        hds = splits[split]

        all_labels, all_img_infos, img_feats = [], [], []
        num_imgs = len(hds)
        for im_id in range(num_imgs):
            im_data = hds.get_img_entry(im_id)  # type: ImgEntry

            im_infos = np.array([*im_data.img_size, im_data.scale])  # size is the original one
            all_img_infos.append(im_infos)

            vout = vm(im_data.image)
            img_feats.append(vout)

            if split != Splits.TEST:
                all_labels.append(im_data.interactions)

            if im_id % 10 == 0 or im_id == num_imgs - 1:
                print('Image %6d/%d' % (im_id, num_imgs))

            torch.cuda.empty_cache()

        img_feats = np.concatenate(img_feats, axis=0)
        num_cached_imgs, feat_dim = img_feats.shape
        assert feat_dim == vm.vis_feat_dim
        assert num_cached_imgs == num_imgs

        all_img_infos = np.stack(all_img_infos, axis=0)
        assert all_img_infos.shape[0] == num_imgs

        precomputed_feats_fn = cfg.program.precomputed_feats_format % ('hico', cfg.model.rcnn_arch, split.value)
        with h5py.File(precomputed_feats_fn, 'w') as feat_file:
            feat_file.create_dataset('img_feats', data=img_feats)
            feat_file.create_dataset('img_infos', data=all_img_infos)

            if all_labels:
                all_labels = np.concatenate(all_labels, axis=0)
                num_cached_imgs, num_interactions = all_labels.shape
                assert num_interactions == hds.hico.num_interactions
                assert num_cached_imgs == num_imgs
                feat_file.create_dataset('labels', data=all_labels.astype(np.int))


if __name__ == '__main__':
    save_feats()
