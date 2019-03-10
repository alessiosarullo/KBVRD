import sys

import h5py
import numpy as np
import torch

from config import cfg
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.dataset.utils import Splits, Minibatch
from lib.models.visual_modules import VisualModule


def save_feats():
    def empty_lists(num, length):
        return ([None for j in range(length)] for i in range(num))
    batch_size = 1
    flipping = False

    sys.argv += ['--model', 'base', '--save_dir', 'fake']  # fake required arguments
    sys.argv += ['--batch_size', str(batch_size)]
    cfg.parse_args()

    train_split = HicoDetInstanceSplit.get_split(split=Splits.TRAIN, flipping_prob=float(flipping))
    val_split = HicoDetInstanceSplit.get_split(split=Splits.VAL, flipping_prob=float(flipping))
    test_split = HicoDetInstanceSplit.get_split(split=Splits.TEST, flipping_prob=float(flipping))

    vm = VisualModule(dataset=train_split)
    vm.cuda()
    vm.eval()
    for split, hds in [(Splits.VAL, val_split),
                       # (Splits.TRAIN, train_split),
                       # (Splits.TEST, test_split),
                       ]:
        hdsl = hds.get_loader(batch_size=batch_size, shuffle=False, drop_last=False)

        precomputed_feats_fn = cfg.program.precomputed_feats_file_format % (cfg.model.rcnn_arch, split.value)
        feat_file = h5py.File(precomputed_feats_fn, 'w')
        feat_file.create_dataset('box_feats', shape=(0, vm.vis_feat_dim), maxshape=(None, vm.vis_feat_dim))
        feat_file.create_dataset('boxes_ext', shape=(0, hds.num_object_classes + 5), maxshape=(None, hds.num_object_classes + 5))
        feat_file.create_dataset('masks', shape=(0, vm.mask_resolution, vm.mask_resolution), maxshape=(None, vm.mask_resolution, vm.mask_resolution))
        feat_file.create_dataset('union_boxes_feats', shape=(0, vm.vis_feat_dim), maxshape=(None, vm.vis_feat_dim))

        try:
            all_union_boxes, all_hoi_infos, all_box_labels, all_hoi_labels = empty_lists(4, len(hdsl))
            box_cache = {k: [] for k in feat_file if not k.startswith('union')}
            hoi_cache = {k: [] for k in feat_file if k.startswith('union')}
            inference = (split == Splits.TEST)
            for im_i, im_data in enumerate(hdsl):
                im_data = im_data  # type: Minibatch
                assert len(im_data.other_ex_data) == 1
                assert im_data.other_ex_data[0]['flipped'] == flipping

                x = vm(im_data, mode_inference=inference)

                x = (value.cpu().numpy() if value is not None and not isinstance(value, np.ndarray) else value for value in x)
                boxes_ext, box_feats, masks, union_boxes, union_boxes_feats, hoi_infos, box_labels, hoi_labels = x

                if boxes_ext is not None:
                    assert np.all(boxes_ext[:, 0] == 0)  # because batch size is 1
                    boxes_ext[:, 0] = im_i
                    box_cache['box_feats'].append(box_feats)
                    box_cache['boxes_ext'].append(boxes_ext)
                    box_cache['masks'].append(masks)

                    if hoi_infos is not None:
                        assert np.all(hoi_infos[:, 0] == 0)

                        hoi_infos[:, 0] = im_i
                        all_hoi_infos[im_i] = hoi_infos
                        all_union_boxes[im_i] = union_boxes
                        hoi_cache['union_boxes_feats'].append(union_boxes_feats)

                        all_box_labels[im_i] = box_labels
                        all_hoi_labels[im_i] = hoi_labels
                else:
                    assert inference

                if im_i % 10 == 0 or im_i == len(hdsl) - 1:
                    print('Image %6d/%d' % (im_i, len(hdsl)))

                    box_cache = {k: np.concatenate(v, axis=0) for k, v in box_cache.items()}
                    num_rois, feat_dim = box_cache['box_feats'].shape
                    assert feat_dim == vm.vis_feat_dim
                    assert all([v.shape[0] == num_rois for v in box_cache.values()])
                    if num_rois > 0:
                        for k, v in box_cache.items():
                            feat_file[k].resize(feat_file[k].shape[0] + num_rois, axis=0)
                            feat_file[k][-num_rois:, :] = v
                    box_cache = {k: [] for k in box_cache.keys()}

                    hoi_cache = {k: np.concatenate(v, axis=0) for k, v in hoi_cache.items()}
                    num_hois = hoi_cache['union_boxes_feats'].shape[0]
                    if num_hois > 0:
                        for k, v in hoi_cache.items():
                            feat_file[k].resize(feat_file[k].shape[0] + num_hois, axis=0)
                            feat_file[k][-num_hois:, :] = v
                    hoi_cache = {k: [] for k in hoi_cache.keys()}

                    torch.cuda.empty_cache()

            all_union_boxes = np.concatenate([x for x in all_union_boxes if x is not None], axis=0)
            all_hoi_infos = np.concatenate([x for x in all_hoi_infos if x is not None], axis=0)
            feat_file.create_dataset('image_ids', data=np.array(hds.image_ids, dtype=np.int))
            feat_file.create_dataset('union_boxes', data=all_union_boxes)
            feat_file.create_dataset('hoi_infos', data=all_hoi_infos.astype(np.int))

            if any([b is not None for b in all_box_labels]):
                assert all([b is not None for b in all_box_labels])
                assert all([b is not None for b in all_hoi_labels])
                all_box_labels = np.concatenate([x for x in all_box_labels if x is not None], axis=0)
                all_hoi_labels = np.concatenate([x for x in all_hoi_labels if x is not None], axis=0)
                feat_file.create_dataset('box_labels', data=all_box_labels.astype(np.int))
                feat_file.create_dataset('hoi_labels', data=all_hoi_labels)
            else:
                assert not any([b is not None for b in all_hoi_labels])
            assert feat_file['union_boxes_feats'].shape[0] == all_union_boxes.shape[0] == all_hoi_infos.shape[0], \
                (feat_file['union_boxes_feats'].shape[0], all_union_boxes.shape[0], all_hoi_infos.shape[0])
        finally:
            feat_file.close()
            print('%s feat file closed.' % split.value)


if __name__ == '__main__':
    save_feats()
