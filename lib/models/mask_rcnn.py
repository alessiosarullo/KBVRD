import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from config import Configs as cfg
from scripts.utils import Timer
from lib.dataset.hicodet import HicoDetSplit, Splits, Minibatch
from lib.pydetectron_integration.detection import im_detect_all_with_feats
from lib.pydetectron_integration.wrappers import segm_results, dummy_datasets, Generalized_RCNN, vis_utils, load_detectron_weight


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
            batch_tensor = x.imgs
            orig_im_infos = x.img_infos

            im_scales = orig_im_infos[:, 2]
            im_infos = np.concatenate([np.tile(batch_tensor.shape[2:], reps=[im_scales.size, 1]), im_scales[:, None]], axis=1)
            inputs = {'data': batch_tensor,
                      'im_info': torch.Tensor(im_infos), }
            Timer.get('Epoch', 'Batch', 'Detect').tic()
            box_class_scores, boxes, box_classes, box_im_ids, masks, feat_map, scores = im_detect_all_with_feats(self.mask_rcnn, inputs)
            Timer.get('Epoch', 'Batch', 'Detect').toc()
            if kwargs.get('return_det_results', False):
                return box_class_scores, boxes, box_classes, box_im_ids, masks, feat_map

            # pick the mask corresponding to the predicted class and binarize it
            masks = torch.stack([masks[i, c, :, :].round() for i, c in enumerate(box_classes)], dim=0)

            boxes_ext = np.concatenate([box_im_ids[:, None].astype(np.float32, copy=False),
                                        boxes.astype(np.float32, copy=False),
                                        scores.astype(np.float32, copy=False)
                                        ], axis=1)
            box_feats = self.get_rois_feats(feat_map, rois=boxes_ext[:, :5])  # FIXME seems like a waste to repeat it
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


def main():
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

        scores, boxes, box_classes, im_ids, masks, feat_map = mask_rcnn(batch, return_det_results=True)

        boxes_with_scores = np.concatenate([boxes, scores[:, None]], axis=1)
        print(boxes)
        print(scores)
        print()
        continue

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

    import pickle
    with open('cache/tmp_mask_result.pkl', 'wb') as f:
        pickle.dump((scores, boxes, box_classes, im_ids, feat_map), f)


if __name__ == '__main__':
    main()
