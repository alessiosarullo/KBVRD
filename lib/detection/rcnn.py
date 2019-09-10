import numpy as np
import torch
import torch.nn as nn

from config import cfg
from lib.detection.detection import im_detect_boxes, get_rois_feats
from lib.detection.wrappers import Generalized_RCNN, load_detectron_weight


class GenRCNN(nn.Module):
    def __init__(self):
        super().__init__()

        if not torch.cuda.is_available():
            raise ValueError("Need a CUDA device to run the code.")

        self.output_feat_dim = cfg.rcnn_output_dim
        self.rcnn = Generalized_RCNN()
        self.allow_train = False

        weight_file = cfg.detectron_pretrained_file_format % cfg.rcnn_arch
        print("Loading RCNN's weights from {}.".format(weight_file))
        load_detectron_weight(self.rcnn, weight_file)

        for param in self.parameters():
            param.requires_grad = self.allow_train

    def train(self, mode=True):
        super().train(mode=self.allow_train and mode)

    def forward(self, image, **kwargs):
        """
        :return: - scores [array]
                 - boxes [array]
                 - box_classes [array] NOTE: classes here include BG one, not present in HICO
                 - box_im_ids [array]
                 - masks [tensor]
                 - feat_map [tensor]
        """

        assert self.allow_train or not self.training
        with torch.set_grad_enabled(self.training):
            assert torch.cuda.is_available()

            boxes, scores, fmap, im_scale = im_detect_boxes(self.rcnn, img=image)

            boxes = boxes * im_scale
            boxes_ext = np.concatenate([
                np.zeros(boxes.shape[0]),  # these were image ids, but now only one image is supported.
                boxes.astype(np.float32, copy=False),
                scores.astype(np.float32, copy=False)
            ], axis=1)

            # Checks
            im_size = np.array([[image.shape[1], image.shape[0]]]).astype(np.float32) * im_scale
            norm_boxes = boxes_ext[:, 1:5] / np.tile(im_size, [1, 2])
            assert np.all(0 <= norm_boxes), (boxes_ext[:, 1:5], im_size, norm_boxes)
            assert np.all(norm_boxes <= 1), (boxes_ext[:, 1:5], im_size, norm_boxes)

            im_area = np.prod(im_size, axis=1)
            box_widths = boxes_ext[:, 3] - boxes_ext[:, 1]
            box_heights = boxes_ext[:, 4] - boxes_ext[:, 2]
            norm_box_areas = box_widths * box_heights / im_area
            assert np.all(0 < norm_box_areas), (boxes_ext[:, 1:5], im_size, norm_boxes)
            assert np.all(norm_box_areas <= 1), (boxes_ext[:, 1:5], im_size, norm_boxes)

            return boxes_ext, im_scale, fmap.detach()

    def get_rois_feats(self, fmap, rois):
        if rois.size == 0:
            rois_feats = fmap.new_zeros((0, self.output_feat_dim))
        else:
            rois_feats = get_rois_feats(self.rcnn, fmap, rois.astype(np.float32, copy=False))
        return rois_feats.detach()
