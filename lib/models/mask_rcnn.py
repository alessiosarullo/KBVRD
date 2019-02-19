import os
import sys

import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn

from config import Configs as cfg
from lib.dataset.hicodet import HicoDetSplit, Splits
from lib.dataset.minibatch import Minibatch
from lib.pydetectron_integration.detection import im_detect_all_with_feats, im_detect_mask
from lib.pydetectron_integration.wrappers import segm_results, dummy_datasets, Generalized_RCNN, vis_utils, load_detectron_weight
from scripts.utils import Timer


class MaskRCNN(nn.Module):
    """
    Wrapper around Detectron's Mask-RCNN
    """

    def __init__(self):
        super().__init__()

        if not torch.cuda.is_available():
            raise ValueError("Need a CUDA device to run the code.")

        mask_rcnn = Generalized_RCNN()
        weight_file = cfg.program.detectron_pretrained_file_format % cfg.model.rcnn_arch
        print("Loading Mask-RCNN's weights from {}.".format(weight_file))
        load_detectron_weight(mask_rcnn, weight_file)

        self.mask_rcnn = mask_rcnn
        self.mask_resolution = cfg.model.mask_resolution
        self.output_feat_dim = 2048  # this is hardcoded in `ResNet_roi_conv5_head_for_masks()`, so I can't actually read it from configs

    def train(self, mode=True):
        super().train(mode=False)  # FIXME freeze weights as well

    def forward(self, x, **kwargs):
        with torch.set_grad_enabled(self.training):
            return self._forward(x, **kwargs)

    def _forward(self, x: Minibatch, **kwargs):
        """
        :param x:
        :param kwargs:
        :return: - scores [array]:
                 - boxes [array]:
                 - box_classes [array]: NOTE: classes here include BG one, not present in HICO
                 - box_im_ids [array]:
                 - masks [tensor]:
                 - feat_map [tensor]:
        """
        # TODO docs

        assert not self.training
        apply_head_only = kwargs.get('head_only', False)

        if not apply_head_only:
            inputs = {'data': x.imgs,
                      'im_info': x.img_infos, }

            if x.pc_box_feats is not None:
                box_feats = x.pc_box_feats
                boxes = x.pc_boxes
                scores = x.pc_box_scores
                box_im_ids = x.box_im_ids
                box_pred_classes = x.box_pred_classes
                box_classes = box_pred_classes[:, 0].int()
                box_class_scores = box_pred_classes[:, 1]

                feat_map = self.mask_rcnn.Conv_Body(inputs)
                masks = im_detect_mask(self.mask_rcnn, inputs['im_info'], box_im_ids, boxes, feat_map)
            else:
                Timer.get('Epoch', 'Batch', 'Detect').tic()
                det_results = im_detect_all_with_feats(self.mask_rcnn, inputs)
                box_class_scores, boxes, box_classes, box_im_ids, masks, feat_map, box_feats, scores = det_results
                Timer.get('Epoch', 'Batch', 'Detect').toc()
            if kwargs.get('return_det_results', False):
                return box_class_scores, boxes, box_classes, box_im_ids, masks, feat_map, box_feats, scores

            # pick the mask corresponding to the predicted class and binarize it
            masks = torch.stack([masks[i, c, :, :].round() for i, c in enumerate(box_classes)], dim=0)

            boxes_ext = np.concatenate([box_im_ids[:, None].astype(np.float32, copy=False),
                                        boxes.astype(np.float32, copy=False),
                                        scores.astype(np.float32, copy=False)
                                        ], axis=1)
            return boxes_ext, masks, feat_map, box_feats
        else:
            try:
                fmap = kwargs['fmap']
                rois = kwargs['rois']
            except KeyError:
                raise
            rois_feats = self.get_rois_feats(fmap, rois)
            return rois_feats

    def get_rois_feats(self, fmap, rois):
        # Input to Box_Head should be a dictionary with the field 'rois' as a Bx5 NumPy array, where each row is [im_id, x1, y1, x2, y2]
        rois_feats = self.mask_rcnn.Box_Head(fmap, {'rois': rois})
        assert all([s == 1 for s in rois_feats.shape[2:]])
        rois_feats.squeeze_(dim=3)
        rois_feats.squeeze_(dim=2)
        return rois_feats


def vis_masks():
    output_dir = 'detectron_outputs/test_ds/'

    batch_size = 2
    num_images = 8

    cfg.parse_args()

    hds = HicoDetSplit(Splits.TEST, im_inds=list(range(num_images)))
    hdsl = hds.get_loader(batch_size=batch_size)
    dummy_coco = dummy_datasets.get_coco_dataset()  # this is used for class names

    mask_rcnn = MaskRCNN()  # add BG class
    mask_rcnn.cuda()
    mask_rcnn.eval()

    for batch_i, batch in enumerate(hdsl):
        print('Batch', batch_i)
        batch = batch  # type: Minibatch

        box_class_scores, boxes, box_classes, im_ids, masks, feat_map, box_feats, scores = mask_rcnn(batch, return_det_results=True)

        boxes_with_scores = np.concatenate([boxes, box_class_scores[:, None]], axis=1)

        masks = masks.cpu().numpy()
        for i in np.unique(im_ids):
            binmask_i = (im_ids == i)
            boxes_with_scores_i = boxes_with_scores[binmask_i, :]
            box_classes_i = box_classes[binmask_i]
            masks_i = masks[binmask_i]

            im_fn = batch.img_fns[i]
            im = cv2.imread(os.path.join(hds.img_dir, im_fn))

            cls_boxes = [[]] + [boxes_with_scores_i[box_classes_i == j, :] for j in range(1, 81)]  # 81 = #coco classes + background
            cls_segms = segm_results(cls_boxes, masks_i, boxes_with_scores_i[:, :-1], im.shape[0], im.shape[1])

            vis_utils.vis_one_image(
                im[:, :, [2, 1, 0]],  # BGR -> RGB for visualization
                os.path.splitext(batch.img_fns[i])[0],
                output_dir,
                cls_boxes,
                cls_segms,
                None,
                dataset=dummy_coco,
                box_alpha=0.3,
                show_class=True,
                thresh=0.7,
                kp_thresh=2
            )


def save_feats():
    # TODO move
    batch_size = 1

    sys.argv += ['--batch_size', str(batch_size)]
    cfg.parse_args()

    im_inds = list(range(cfg.program.num_images)) if cfg.program.num_images > 0 else None
    hds = HicoDetSplit(Splits.TRAIN, im_inds=im_inds)
    hdsl = hds.get_loader(batch_size=batch_size, shuffle=False, drop_last=False)

    mask_rcnn = MaskRCNN()
    mask_rcnn.cuda()
    mask_rcnn.eval()

    precomputed_feats_fn = cfg.program.precomputed_feats_file_format % cfg.model.rcnn_arch
    feat_file = h5py.File(precomputed_feats_fn, 'w')
    feat_file.create_dataset('box_feats', shape=(0, mask_rcnn.output_feat_dim), maxshape=(None, mask_rcnn.output_feat_dim))
    feat_file.create_dataset('box_scores', shape=(0, hds.num_object_classes + 1), maxshape=(None, hds.num_object_classes + 1))  # FIXME magic const
    # feat_file.create_dataset('feat_maps', shape=(0, 1024, 7, 7), maxshape=(None, 1024, 7, 7))  # FIXME check

    try:
        all_boxes, box_im_ids, all_pred_classes = [], [], []
        cached_feats, cached_fmaps, cached_scores = [], [], []
        for im_i, im_data in enumerate(hdsl):
            print('Batch', im_i)
            im_data = im_data  # type: Minibatch

            box_class_scores, boxes, box_classes, im_ids_in_batch, masks, feat_map, box_feats, scores = mask_rcnn(im_data, return_det_results=True)
            assert np.all(im_ids_in_batch == 0)  # because batch size is 1
            im_ids = np.full((boxes.shape[0], 1), fill_value=im_i)
            pred_classes = np.stack([box_classes.astype(np.float, copy=False), box_class_scores], axis=1)
            box_feats = box_feats.cpu().numpy()

            box_im_ids.append(im_ids)
            all_boxes.append(boxes)
            all_pred_classes.append(pred_classes)

            cached_feats.append(box_feats)
            cached_fmaps.append(feat_map)
            cached_scores.append(scores)
            if im_i % 1000 == 0 or im_i == len(hdsl) - 1:
                cached_feats = np.concatenate(cached_feats, axis=0)
                cached_scores = np.concatenate(cached_scores, axis=0)
                # cached_fmaps = np.concatenate(cached_fmaps, axis=0)

                num_rois, feat_dim = cached_feats.shape
                assert feat_dim == mask_rcnn.output_feat_dim
                assert num_rois == cached_scores.shape[0]

                if num_rois > 0:
                    feat_file['box_feats'].resize(feat_file['box_feats'].shape[0] + num_rois, axis=0)
                    feat_file['box_feats'][-num_rois:, :] = cached_feats

                    feat_file['box_scores'].resize(feat_file['box_scores'].shape[0] + num_rois, axis=0)
                    feat_file['box_scores'][-num_rois:, :] = cached_scores

                    # feat_file['feat_maps'].resize(feat_file['feat_maps'].shape[0] + num_rois, axis=0)
                    # feat_file['feat_maps'][-num_rois:, :] = cached_fmaps
                cached_feats, cached_fmaps, cached_scores = [], [], []

        all_boxes = np.concatenate(all_boxes, axis=0)
        box_im_ids = np.concatenate(box_im_ids)
        all_pred_classes = np.concatenate(all_pred_classes, axis=0)
        feat_file.create_dataset('boxes', data=all_boxes)
        feat_file.create_dataset('box_im_ids', data=box_im_ids.astype(np.int, copy=False))
        feat_file.create_dataset('box_pred_classes', data=all_pred_classes)
        feat_file.create_dataset('image_index', data=np.array(hds.image_index, dtype=np.int))
        assert feat_file['box_feats'].shape[0] == all_boxes.shape[0] == all_pred_classes.shape[0], \
            (feat_file['box_feats'].shape[0], all_boxes.shape[0], all_pred_classes.shape[0])
        # assert feat_file['feat_maps'].shape[0] == feat_file['image_index'].shape[0], \
        #     (feat_file['feat_maps'].shape[0], feat_file['image_index'].shape[0])
    finally:
        feat_file.close()


if __name__ == '__main__':
    # vis_masks()
    save_feats()
