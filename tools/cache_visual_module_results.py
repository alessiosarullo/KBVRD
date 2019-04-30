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

    sys.argv += ['--img_batch_size', str(batch_size), '--val_ratio', '0']
    cfg.parse_args(allow_required=False)

    train_split = HicoDetInstanceSplit.get_split(split=Splits.TRAIN, flipping_prob=float(flipping), load_precomputed=False)
    test_split = HicoDetInstanceSplit.get_split(split=Splits.TEST, flipping_prob=float(flipping), load_precomputed=False)
    assert Splits.VAL not in HicoDetInstanceSplit._splits

    vm = VisualModule(dataset=train_split)
    if torch.cuda.is_available():
        vm.cuda()
    else:
        print('!!!!!!!!!!!!!!!!! Running on CPU!')
    vm.eval()
    for split, hds in [(Splits.TRAIN, train_split),
                       (Splits.TEST, test_split),
                       ]:
        hd_loader = hds.get_loader(batch_size=batch_size, shuffle=False, drop_last=False)

        precomputed_feats_fn = cfg.program.precomputed_feats_file_format % (cfg.model.rcnn_arch, split.value)
        feat_file = h5py.File(precomputed_feats_fn, 'w')
        feat_file.create_dataset('box_feats', shape=(0, vm.vis_feat_dim), maxshape=(None, vm.vis_feat_dim))
        feat_file.create_dataset('boxes_ext', shape=(0, hds.num_object_classes + 5), maxshape=(None, hds.num_object_classes + 5))
        feat_file.create_dataset('masks', shape=(0, vm.mask_resolution, vm.mask_resolution), maxshape=(None, vm.mask_resolution, vm.mask_resolution))
        feat_file.create_dataset('union_boxes_feats', shape=(0, vm.vis_feat_dim), maxshape=(None, vm.vis_feat_dim))

        try:
            all_union_boxes, all_ho_infos, all_box_labels, all_action_labels, all_img_infos = empty_lists(5, len(hd_loader))
            obj_cache = {k: [] for k in feat_file if not k.startswith('union')}
            ho_cache = {k: [] for k in feat_file if k.startswith('union')}
            inference = (split == Splits.TEST)
            for im_i, im_data in enumerate(hd_loader):
                im_data = im_data  # type: Minibatch
                assert len(im_data.other_ex_data) == 1
                assert im_data.other_ex_data[0]['flipped'] == flipping

                im_infos = np.array([*im_data.other_ex_data[0]['im_size'], im_data.other_ex_data[0]['im_scale']])
                all_img_infos[im_i] = im_infos

                x = vm(im_data, inference)

                x = (value.cpu().numpy() if value is not None and not isinstance(value, np.ndarray) else value for value in x)
                boxes_ext, box_feats, masks, union_boxes, union_boxes_feats, ho_infos, box_labels, action_labels, _ = x

                if boxes_ext is not None:
                    assert np.all(boxes_ext[:, 0] == 0)  # because batch size is 1
                    boxes_ext[:, 0] = im_i
                    obj_cache['box_feats'].append(box_feats)
                    obj_cache['boxes_ext'].append(boxes_ext)
                    obj_cache['masks'].append(masks)

                    if ho_infos is not None:
                        assert np.all(ho_infos[:, 0] == 0)

                        ho_infos[:, 0] = im_i
                        all_ho_infos[im_i] = ho_infos
                        all_union_boxes[im_i] = union_boxes
                        ho_cache['union_boxes_feats'].append(union_boxes_feats)

                        all_box_labels[im_i] = box_labels
                        all_action_labels[im_i] = action_labels
                else:
                    assert inference

                if im_i % 10 == 0 or im_i == len(hd_loader) - 1:
                    print('Image %6d/%d' % (im_i, len(hd_loader)))

                    obj_cache = {k: np.concatenate(v, axis=0) for k, v in obj_cache.items()}
                    num_rois, feat_dim = obj_cache['box_feats'].shape
                    assert feat_dim == vm.vis_feat_dim
                    assert all([v.shape[0] == num_rois for v in obj_cache.values()])
                    if num_rois > 0:
                        for k, v in obj_cache.items():
                            feat_file[k].resize(feat_file[k].shape[0] + num_rois, axis=0)
                            feat_file[k][-num_rois:, :] = v
                    obj_cache = {k: [] for k in obj_cache.keys()}

                    ho_cache = {k: np.concatenate(v, axis=0) for k, v in ho_cache.items()}
                    num_ho_pairs = ho_cache['union_boxes_feats'].shape[0]
                    if num_ho_pairs > 0:
                        for k, v in ho_cache.items():
                            feat_file[k].resize(feat_file[k].shape[0] + num_ho_pairs, axis=0)
                            feat_file[k][-num_ho_pairs:, :] = v
                    ho_cache = {k: [] for k in ho_cache.keys()}

                    torch.cuda.empty_cache()

            all_union_boxes = np.concatenate([x for x in all_union_boxes if x is not None], axis=0)
            all_ho_infos = np.concatenate([x for x in all_ho_infos if x is not None], axis=0)

            assert len(all_img_infos) == len(hd_loader)
            all_img_infos = np.stack([x for x in all_img_infos if x is not None], axis=0)
            assert all_img_infos.shape[0] == len(hd_loader)

            feat_file.create_dataset('image_ids', data=np.array(hds.image_ids, dtype=np.int))
            feat_file.create_dataset('union_boxes', data=all_union_boxes)
            feat_file.create_dataset('ho_infos', data=all_ho_infos.astype(np.int))
            feat_file.create_dataset('img_infos', data=all_img_infos)

            if any([b is not None for b in all_box_labels]):
                assert all([b is not None for b in all_box_labels])
                assert all([b is not None for b in all_action_labels])
                all_box_labels = np.concatenate([x for x in all_box_labels if x is not None], axis=0)
                all_action_labels = np.concatenate([x for x in all_action_labels if x is not None], axis=0)
                feat_file.create_dataset('box_labels', data=all_box_labels.astype(np.int))
                feat_file.create_dataset('action_labels', data=all_action_labels)
            else:
                assert not any([b is not None for b in all_action_labels])
            assert feat_file['union_boxes_feats'].shape[0] == all_union_boxes.shape[0] == all_ho_infos.shape[0], \
                (feat_file['union_boxes_feats'].shape[0], all_union_boxes.shape[0], all_ho_infos.shape[0])
        finally:
            feat_file.close()
            print('%s feat file closed.' % split.value)


if __name__ == '__main__':
    save_feats()
