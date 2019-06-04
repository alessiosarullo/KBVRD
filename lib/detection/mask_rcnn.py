import numpy as np
import torch
import torch.nn as nn

from config import cfg
from lib.dataset.hicodet.hicodet_split import HicoDetSplitBuilder, Minibatch
from lib.detection.detection import im_detect_boxes, im_detect_mask, get_rois_feats
from lib.detection.wrappers import Generalized_RCNN, load_detectron_weight


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

    def forward(self, x: Minibatch, **kwargs):
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
        with torch.set_grad_enabled(self.training):
            box_im_ids, boxes, scores, fmap = im_detect_boxes(self.mask_rcnn, inputs={'data': x.imgs, 'im_info': x.img_infos})
            im_scales = x.img_infos[:, 2].cpu().numpy()
            boxes = boxes * im_scales[box_im_ids, None]  # Note: this can introduce numerical errors! Box might be bigger than image.
            boxes_ext = np.concatenate([box_im_ids[:, None].astype(np.float32, copy=False),
                                        boxes.astype(np.float32, copy=False),
                                        scores.astype(np.float32, copy=False)
                                        ], axis=1)
            return boxes_ext, fmap.detach()

    def get_rois_feats(self, fmap, rois):
        if rois.size == 0:
            rois_feats = fmap.new_zeros((0, self.output_feat_dim))
        else:
            rois_feats = get_rois_feats(self.mask_rcnn, fmap, rois.astype(np.float32, copy=False))
        return rois_feats.detach()

    def get_masks(self, fmap, rois, box_classes):
        """
        Detects a mask for each class, then pick the one corresponding to the given class and binarize it.
        :param fmap:
        :param rois:
        :param box_classes:
        :return:
        """
        if rois.size == 0:
            masks = fmap.new_zeros((0, self.mask_resolution, self.mask_resolution))
        else:
            box_im_ids = rois[:, 0].astype(np.int, copy=False)
            boxes = rois[:, 1:5].astype(np.float32, copy=False)
            masks = im_detect_mask(self.mask_rcnn, box_im_ids, boxes, fmap)
            masks = torch.stack([masks[i, c, :, :] for i, c in enumerate(box_classes)], dim=0)
        return masks.detach()
