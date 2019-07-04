import sys

import h5py
import numpy as np
import torch
import torch.utils.data

from config import cfg
from lib.dataset.hicodet.hicodet_split import HicoDetSplitBuilder, HicoDetSplit, Minibatch
from lib.dataset.utils import Splits
from lib.detection.visual_module import VisualModule, VisualOutput


def save_feats():
    def empty_lists(num, length):
        return ([None for j in range(length)] for i in range(num))

    sys.argv += ['--img_batch_size', '1', '--val_ratio', '0']
    cfg.parse_args(fail_if_missing=False)

    train_split = HicoDetSplitBuilder.get_split(HicoDetSplit, split=Splits.TRAIN)
    test_split = HicoDetSplitBuilder.get_split(HicoDetSplit, split=Splits.TEST)
    assert Splits.VAL not in HicoDetSplitBuilder.splits[HicoDetSplit]

    vm = VisualModule(dataset=train_split)
    if torch.cuda.is_available():
        vm.cuda()
    else:
        print('!!!!!!!!!!!!!!!!! Running on CPU!')
    vm.eval()
    for split, hds in [(Splits.TRAIN, train_split),
                       (Splits.TEST, test_split),
                       ]:
        hd_loader = torch.utils.data.DataLoader(dataset=hds,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=0,
                                                collate_fn=lambda x: Minibatch.collate(x),
                                                drop_last=False,
                                                )

        precomputed_feats_fn = cfg.program.precomputed_feats_file_format % (cfg.model.rcnn_arch, split.value)
        feat_file = h5py.File(precomputed_feats_fn, 'w')
        feat_file.create_dataset('box_feats', shape=(0, vm.vis_feat_dim), maxshape=(None, vm.vis_feat_dim))
        feat_file.create_dataset('boxes_ext', shape=(0, hds.num_object_classes + 5), maxshape=(None, hds.num_object_classes + 5))
        feat_file.create_dataset('union_boxes_feats', shape=(0, vm.vis_feat_dim), maxshape=(None, vm.vis_feat_dim))

        try:
            all_union_boxes, all_ho_infos, all_box_labels, all_action_labels, all_img_infos = empty_lists(5, len(hd_loader))
            obj_cache = {k: [] for k in feat_file if not k.startswith('union')}
            ho_cache = {k: [] for k in feat_file if k.startswith('union')}
            inference = (split == Splits.TEST)
            for im_i, im_data in enumerate(hd_loader):
                # if im_i != 1951:
                #     continue
                im_data = im_data  # type: Minibatch
                assert len(im_data.other_ex_data) == 1

                im_infos = np.array([*im_data.other_ex_data[0]['im_size'], im_data.other_ex_data[0]['im_scale']])
                all_img_infos[im_i] = im_infos

                vout = vm(im_data, inference)  # type: VisualOutput
                boxes_ext = vout.boxes_ext
                box_feats = vout.box_feats
                ho_infos = vout.ho_infos_np
                union_boxes = vout.hoi_union_boxes
                union_boxes_feats = vout.hoi_union_boxes_feats
                box_labels = vout.box_labels
                action_labels = vout.action_labels
                if boxes_ext is not None:
                    boxes_ext = boxes_ext.cpu().numpy()
                    box_feats = box_feats.cpu().numpy()
                if ho_infos is not None:
                    union_boxes_feats = union_boxes_feats.cpu().numpy()
                if box_labels is not None:
                    box_labels = box_labels.cpu().numpy()
                if action_labels is not None:
                    action_labels = action_labels.cpu().numpy()

                if boxes_ext is not None:
                    assert np.all(boxes_ext[:, 0] == 0)  # because batch size is 1
                    boxes_ext[:, 0] = im_i
                    obj_cache['box_feats'].append(box_feats)
                    obj_cache['boxes_ext'].append(boxes_ext)

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
