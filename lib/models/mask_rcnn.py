import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from config import Configs as cfg
from lib.dataset.hicodet import HicoDetSplit, Splits
from lib.dataset.minibatch import Minibatch
from lib.detection.detection import im_detect_all_with_feats, im_detect_mask, get_rois_feats
from lib.detection.wrappers import segm_results, dummy_datasets, Generalized_RCNN, vis_utils, load_detectron_weight
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
            if x.pc_box_feats is not None:
                box_feats = x.pc_box_feats
                boxes = x.pc_boxes
                scores = x.pc_box_scores
                box_im_ids = x.box_im_ids
                box_pred_classes = x.box_pred_classes
                box_classes = box_pred_classes[:, 0].astype(np.int)
                box_class_scores = box_pred_classes[:, 1]

                Timer.get('Epoch', 'Batch', 'Conv').tic()
                feat_map = self.mask_rcnn.Conv_Body(x.imgs)
                Timer.get('Epoch', 'Batch', 'Conv').toc(synchronize=True)
                Timer.get('Epoch', 'Batch', 'Mask').tic()
                masks = im_detect_mask(self.mask_rcnn, x.img_infos, box_im_ids, boxes, feat_map)
                Timer.get('Epoch', 'Batch', 'Mask').toc(synchronize=True)
            else:
                inputs = {'data': x.imgs,
                          'im_info': x.img_infos, }
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
            rois_feats = get_rois_feats(self.mask_rcnn, fmap, rois)
            return rois_feats


def vis_masks():
    cfg.parse_args()
    output_dir = os.path.join('output', 'tmp', '%s' % ('pre' if cfg.program.load_precomputed_feats else 'e2e'))

    im_inds = list(range(cfg.program.num_images)) if cfg.program.num_images > 0 else None
    hds = HicoDetSplit(Splits.TRAIN, im_inds=im_inds)
    hdsl = hds.get_loader(batch_size=cfg.opt.batch_size, shuffle=False)
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


if __name__ == '__main__':
    vis_masks()
