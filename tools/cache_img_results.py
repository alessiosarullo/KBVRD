import sys

import h5py
import numpy as np
import torch
import torch.utils.data
from config import cfg
from lib.dataset.hico.hico_split import HicoSplit
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
        device = torch.device('cuda')
    else:
        print('!!!!!!!!!!!!!!!!! Running on CPU!')
        device = torch.device('cpu')
    vm.eval()
    for split in [Splits.TRAIN, Splits.TEST]:
        hds = splits[split]

        all_img_feats = []
        num_imgs = len(hds)
        hds_data_loader = hds.get_img_loader()
        for im_id, img in enumerate(hds_data_loader):
            feats = vm(img.to(device=device))
            all_img_feats.append(feats)
            if im_id % 100 == 0 or im_id == num_imgs - 1:
                print('Image %6d/%d' % (im_id, num_imgs))
                torch.cuda.empty_cache()

        all_img_feats = np.concatenate(all_img_feats, axis=0)
        num_cached_imgs, feat_dim = all_img_feats.shape
        assert feat_dim == vm.vis_feat_dim
        assert num_cached_imgs == num_imgs

        precomputed_feats_fn = cfg.program.precomputed_feats_format % ('hico', cfg.model.rcnn_arch, split.value)
        with h5py.File(precomputed_feats_fn, 'w') as feat_file:
            feat_file.create_dataset('img_feats', data=all_img_feats)


if __name__ == '__main__':
    save_feats()
