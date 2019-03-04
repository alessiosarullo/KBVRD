import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from config import Configs as cfg
from lib.containers import Minibatch
from lib.dataset.hicodet import HicoDetInstanceSplit, Splits
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

        self.mask_resolution = cfg.model.mask_resolution
        self.output_feat_dim = 2048  # this is hardcoded in `ResNet_roi_conv5_head_for_masks()`, so I can't actually read it from configs
        self.mask_rcnn = Generalized_RCNN()
        self._load_weights()
        self.allow_train = False

        for param in self.parameters():
            param.requires_grad = self.allow_train

    def _load_weights(self):
        weight_file = cfg.program.detectron_pretrained_file_format % cfg.model.rcnn_arch
        print("Loading Mask-RCNN's weights from {}.".format(weight_file))
        load_detectron_weight(self.mask_rcnn, weight_file)

    def train(self, mode=True):
        super().train(mode=self.allow_train and mode)

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

        assert self.allow_train or not self.training

        if x.pc_box_feats is not None:
            box_feats = x.pc_box_feats
            boxes = x.pc_boxes
            scores = x.pc_box_scores
            box_im_ids = x.box_im_ids
            box_pred_classes = x.box_pred_classes
            box_classes = box_pred_classes[:, 0].astype(np.int)
            box_class_scores = box_pred_classes[:, 1]

            fmap = self.mask_rcnn.Conv_Body(x.imgs)
            masks = self.get_masks(x.img_infos, fmap, boxes, box_im_ids)
        else:
            inputs = {'data': x.imgs,
                      'im_info': x.img_infos
                      }
            box_class_scores, boxes, box_classes, box_im_ids, masks, fmap, box_feats, scores = im_detect_all_with_feats(self.mask_rcnn, inputs)
        if kwargs.get('return_det_results', False):
            im_scales = x.img_infos[:, 2].cpu().numpy()
            boxes /= im_scales[box_im_ids, None]
            return box_class_scores, boxes, box_classes, box_im_ids, masks, fmap, box_feats, scores

        # pick the mask corresponding to the predicted class and binarize it
        masks = torch.stack([masks[i, c, :, :].round() for i, c in enumerate(box_classes)], dim=0)

        boxes_ext = np.concatenate([box_im_ids[:, None].astype(np.float32, copy=False),
                                    boxes.astype(np.float32, copy=False),
                                    scores.astype(np.float32, copy=False)
                                    ], axis=1)
        fmap = fmap.detach()
        box_feats = box_feats.detach()
        masks = masks.detach()
        return boxes_ext, masks, fmap, box_feats

    def get_rois_feats(self, fmap, rois):
        return get_rois_feats(self.mask_rcnn, fmap, rois.astype(np.float32, copy=False)).detach()

    def get_masks(self, img_infos, fmap, boxes, box_im_ids, box_classes=None):
        box_im_ids = box_im_ids.astype(np.int, copy=False)
        im_scales = img_infos[:, 2].cpu().numpy()
        boxes = boxes * im_scales[box_im_ids, None]
        masks = im_detect_mask(self.mask_rcnn, box_im_ids, boxes.astype(np.float32, copy=False), fmap)
        if box_classes is not None:
            masks = torch.stack([masks[i, c, :, :].round() for i, c in enumerate(box_classes)], dim=0)
        return masks.detach()


def vis_masks():
    cfg.parse_args()
    output_dir = os.path.join('output', 'tmp', '%s' % ('pre' if cfg.program.load_precomputed_feats else 'e2e'))

    # hds = HicoDetSplit(Splits.TRAIN, im_inds=im_inds)
    # hdsl = hds.get_loader(batch_size=cfg.opt.batch_size, shuffle=False)
    hds = HicoDetInstanceSplit.get_split(split=Splits.TEST, im_inds=[0])
    hdsl = hds.get_loader(batch_size=1, shuffle=False)
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

            im_fn = batch.other_ex_data[i]['fn']
            im = cv2.imread(os.path.join(hds.img_dir, im_fn))

            cls_boxes = [[]] + [boxes_with_scores_i[box_classes_i == j, :] for j in range(1, len(dummy_coco.classes))]  # background is included
            cls_segms = segm_results(cls_boxes, masks_i, boxes_with_scores_i[:, :-1], im.shape[0], im.shape[1])
            # cls_segms = None

            vis_utils.vis_one_image(
                im[:, :, [2, 1, 0]],  # BGR -> RGB for visualization
                os.path.splitext(im_fn)[0],
                output_dir,
                cls_boxes,
                cls_segms,
                None,
                dataset=dummy_coco,
                box_alpha=0.3,
                show_class=True,
                thresh=0.7,  # Lower this to see all the predictions (was 0.7 in the original code)
                kp_thresh=2,
                ext='png'
            )


if __name__ == '__main__':
    vis_masks()
