import argparse
import sys
from collections import defaultdict

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.nn.parallel

from drivers.datasets import HicoDetSplit
from models.pydetectron.lib.core.config import cfg
from models.pydetectron.lib.core.test_engine import initialize_model_from_cfg
from models.pydetectron.lib.utils.timer import Timer
from models.pydetectron.lib.utils.vis import convert_from_cls_format
import models.pydetectron.lib.utils.fpn as fpn_utils


class MaskRCNN(nn.Module):
    def __init__(self, dataset: HicoDetSplit, mask_args):
        super().__init__()

        self.dataset = dataset
        self.detectron_maskRCNN = initialize_model_from_cfg(mask_args, gpu_id=mask_args.gpu_id)

    def forward(self, x, **kwargs):
        """
        :param x:  {'imgs': [],
                    'img_size': [],
                    'gt_boxes': [],
                    'gt_box_classes': [],
                    'gt_inters': [],
                    }
        :return:
        """

        with torch.set_grad_enabled(self.training):
            return self._forward(x, **kwargs)

    def _forward(self, x, **kwargs):
        boxes_list, masks_list, img_inds, fmaps = [], [], [], []
        for im_i, (im, img_info) in enumerate(zip(x['img'], x['img_info'])):
            cls_boxes_i, cls_segms_i, fmap = self.im_detect_all(im, img_info)
            boxes_i, segms_i, keyps_i, classes_i = convert_from_cls_format(cls_boxes_i, cls_segms_i, None)

            assert boxes_i is not None and segms_i is not None and boxes_i.shape[0] == segms_i.shape[0].shape[0] > 0
            masks_i = mask_util.decode(segms_i)

            boxes_list.append(boxes_i)
            masks_list.append(masks_i)
            img_inds += [im_i] * boxes_i.shape[0]
            fmaps.append(fmap)

        boxes_w_scores = np.concatenate(boxes_list, axis=0)
        masks = np.concatenate(masks_list, axis=0)
        img_inds = np.array(img_inds)
        fmaps = np.stack(fmaps, axis=0)

        self.proposal_assignments_det(boxes)

    def im_detect_all(self, im, img_info):
        model = self.detectron_maskRCNN
        timers = defaultdict(Timer)

        timers['im_detect_bbox'].tic()
        assert not cfg.TEST.BBOX_AUG.ENABLED
        scores, boxes, blob_conv = self.im_detect_bbox(model, im, img_info)
        timers['im_detect_bbox'].toc()

        assert cfg.MODEL.MASK_ON and boxes.shape[0] > 0
        timers['im_detect_mask'].tic()
        assert not cfg.TEST.MASK_AUG.ENABLED
        masks = self.im_detect_mask(model, img_info[2], boxes, blob_conv)
        timers['im_detect_mask'].toc()

        timers['misc_mask'].tic()
        cls_segms = segm_results(cls_boxes, masks, boxes, im.shape[0], im.shape[1])
        timers['misc_mask'].toc()

        assert not cfg.MODEL.KEYPOINTS_ON

        return cls_boxes, cls_segms, blob_conv

    def im_detect_bbox(self, model, im, img_info):
        assert cfg.MODEL.FASTER_RCNN
        assert not cfg.PYTORCH_VERSION_LESS_THAN_040
        assert cfg.TEST.BBOX_REG
        assert not cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED
        assert not cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

        model_output = model(data=[im], im_info=[img_info])
        boxes = model_output['rois'][:, 1:5]

        scores = model_output['cls_score']  # cls prob (activations after softmax)
        # scores = scores.reshape([-1, scores.shape[-1]])  # In case there is 1 proposal TODO check if needed
        box_deltas = model_output['bbox_pred']
        # box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])  # In case there is 1 proposal

        im_scale = img_info[2]
        boxes = self.bbox_transform(boxes / im_scale, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS, im.shape)  # THIS ARE AT ORIGINAL IMAGE SCALE
        scores, boxes = self.nms_boxes(scores, boxes)
        return scores, boxes, model_output['blob_conv']

    @staticmethod
    def im_detect_mask(model, im_scale, boxes, blob_conv):
        """Infer instance segmentation masks. This function must be called after im_detect_bbox as it assumes that the Caffe2 workspace is already
        populated with the necessary blobs.

        Arguments:
            model (DetectionModelHelper): the detection model to use
            im_scale (list): image blob scales as returned by im_detect_bbox
            boxes (ndarray): R x 4 array of bounding box detections (e.g., as returned by im_detect_bbox)
            blob_conv (Variable): base features from the backbone network.

        Returns:
            pred_masks (ndarray): R x K x M x M array of class specific soft masks output by the network (must be processed by segm_results to
                convert into hard masks in the original image coordinate space)
        """
        assert boxes.shape[0] > 0
        assert not cfg.FPN.MULTILEVEL_ROIS
        assert cfg.MRCNN.CLS_SPECIFIC_MASK
        M = cfg.MRCNN.RESOLUTION
        mask_rois = torch.cat([boxes.new_zeros((boxes.shape[0], 1)), boxes * im_scale], axis=1)
        pred_masks = model.module.mask_net(blob_conv, rpn_blob={'mask_rois': mask_rois})
        pred_masks = pred_masks.view([-1, cfg.MODEL.NUM_CLASSES, M, M])
        return pred_masks

    @staticmethod
    def bbox_transform(boxes: torch.Tensor, deltas: torch.Tensor, weights, im_shape):
        assert boxes.shape[0] > 0

        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into np.exp()
        dw = dw.clamp(max=cfg.BBOX_XFORM_CLIP)
        dh = dh.clamp(max=cfg.BBOX_XFORM_CLIP)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = deltas.new_zeros(deltas.shape)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1  # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1  # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)

        # Clip
        assert pred_boxes.shape[1] % 4 == 0, pred_boxes.shape[1]
        pred_boxes[:, 0::4] = pred_boxes[:, 0::4].clamp(min=0, max=im_shape[1] - 1)
        pred_boxes[:, 1::4] = pred_boxes[:, 1::4].clamp(min=0, max=im_shape[0] - 1)
        pred_boxes[:, 2::4] = pred_boxes[:, 2::4].clamp(min=0, max=im_shape[1] - 1)
        pred_boxes[:, 3::4] = pred_boxes[:, 3::4].clamp(min=0, max=im_shape[0] - 1)
        return pred_boxes

    @staticmethod
    def nms_boxes(obj_dists, rois):
        """
        Performs NMS on the boxes
        :param obj_dists: [#rois, #classes]
        :param rois: [#rois, 4]
        :return
            nms_inds [#nms]
            nms_scores [#nms]
            nms_labels [#nms]
            nms_boxes_assign [#nms, 4]
            nms_boxes  [#nms, #classes, 4]. classid=0 is the box prior.
        """
        # Now produce the boxes
        # box deltas is (num_rois, num_classes, 4) but rois is only #(num_rois, 4)
        boxes = rois.view(rois.shape[0], -1, 4)

        # Clip the boxes and get the best N dets per image.
        inds = rois[:, 0].long().contiguous()
        dets = self.filter_det(obj_dists, boxes, max_per_img=cfg.TEST.DETECTIONS_PER_IM, nms_thresh=cfg.TEST.NMS,
                               nms_filter_duplicates=True,  # multiple detections for same box, but different classes
                               )

        if dets is None:
            print("nothing was detected", flush=True)
            return None
        nms_inds, nms_scores, nms_labels = [torch.cat(x, 0) for x in zip(*dets)]
        twod_inds = nms_inds * boxes.size(1) + nms_labels.data
        nms_boxes_assign = boxes.view(-1, 4)[twod_inds]

        nms_boxes = torch.cat((rois[:, 1:][nms_inds][:, None], boxes[nms_inds][:, 1:]), 1)
        return nms_inds, nms_scores, nms_labels, nms_boxes_assign, nms_boxes, inds[nms_inds]

    @staticmethod
    def filter_det(scores, boxes, max_per_img, nms_thresh, nms_filter_duplicates=True,
                   thresh=0.001, pre_nms_topn=6000, post_nms_topn=300):
        """
        Filters the detections for a single image
        :param scores: [num_rois, num_classes]
        :param boxes: [num_rois, num_classes, 4]. Assumes the boxes have been clamped
        :param max_per_img: Max detections per image
        :param thresh: Threshold for calling it a good box
        :param nms_filter_duplicates: True if we shouldn't allow for mulitple detections of the same box (with different labels)
        :return: A numpy concatenated array with up to 100 detections/img [num_im, x1, y1, x2, y2, score, cls]
        """

        valid_cls = (scores[:, 1:].data.max(0)[0] > thresh).nonzero() + 1
        if valid_cls.dim() == 0:
            return None

        nms_mask = scores.data.clone()
        nms_mask.zero_()

        for c_i in valid_cls.squeeze(1).cpu():
            scores_ci = scores.data[:, c_i]
            boxes_ci = boxes.data[:, c_i]

            keep = apply_nms(scores_ci, boxes_ci, pre_nms_topn=pre_nms_topn, post_nms_topn=post_nms_topn, nms_thresh=nms_thresh)
            nms_mask[:, c_i][keep] = 1

        dists_all = nms_mask * scores.data

        if nms_filter_duplicates:
            scores_pre, labels_pre = dists_all.data.max(1)
            inds_all = scores_pre.nonzero()
            assert inds_all.dim() != 0
            inds_all = inds_all.squeeze(1)

            labels_all = labels_pre[inds_all]
            scores_all = scores_pre[inds_all]
        else:
            nz = nms_mask.nonzero()
            assert nz.dim() != 0
            inds_all = nz[:, 0]
            labels_all = nz[:, 1]
            scores_all = scores.data.view(-1)[inds_all * scores.data.size(1) + labels_all]

        # # Limit to max per image detections
        vs, idx = torch.sort(scores_all, dim=0, descending=True)
        idx = idx[vs > thresh]
        if max_per_img < idx.size(0):
            idx = idx[:max_per_img]

        inds_all = inds_all[idx]
        scores_all = scores_all[idx]
        labels_all = labels_all[idx]

        return inds_all, scores_all, labels_all

    def proposal_assignments_det(self, rpn_rois, gt_boxes, gt_classes, fg_thresh=0.5):
        """
        Assign object detection proposals to ground-truth targets. Produces proposal
        classification labels and bounding-box regression targets.
        :param rpn_rois: [img_ind, x1, y1, x2, y2]
        :param gt_boxes:   [num_boxes, 4] array of x0, y0, x1, y1
        :param gt_classes: [num_boxes, 2] array of [img_ind, class]
        :param Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
        :return:
            rois: [num_rois, 5]
            labels: [num_rois] array of labels
            bbox_targets [num_rois, 4] array of targets for the labels.
        """
        fg_rois_per_image = int(np.round(ROIS_PER_IMG * FG_FRACTION))

        gt_img_inds = gt_classes[:, 0]

        all_boxes = torch.cat([rpn_rois[:, 1:], gt_boxes], dim=0)

        ims_per_box = torch.cat([rpn_rois[:, 0].long(), gt_img_inds], dim=0)

        im_sorted, idx = torch.sort(ims_per_box, 0)
        all_boxes = all_boxes[idx]

        # Assume that the GT boxes are already sorted in terms of image id
        num_images = int(im_sorted[-1]) + 1

        labels = []
        rois = []
        bbox_targets = []
        for im_ind in range(num_images):
            g_inds = (gt_img_inds == im_ind).nonzero()

            if g_inds.dim() == 0:
                continue
            g_inds = g_inds.squeeze(1)
            g_start = g_inds[0]
            g_end = g_inds[-1] + 1

            t_inds = (im_sorted == im_ind).nonzero().squeeze(1)
            t_start = t_inds[0]
            t_end = t_inds[-1] + 1

            # Max overlaps: for each predicted box, get the max ROI
            # Get the indices into the GT boxes too (must offset by the box start)
            ious = bbox_overlaps(all_boxes[t_start:t_end], gt_boxes[g_start:g_end])
            max_overlaps, gt_assignment = ious.max(1)
            max_overlaps = max_overlaps.cpu().numpy()
            # print("Best overlap is {}".format(max_overlaps.max()))
            # print("\ngt assignment is {} while g_start is {} \n ---".format(gt_assignment, g_start))
            gt_assignment += g_start

            keep_inds_np, num_fg = _sel_inds(max_overlaps, fg_thresh, fg_rois_per_image,
                                             ROIS_PER_IMG)

            if keep_inds_np.size == 0:
                continue

            keep_inds = torch.LongTensor(keep_inds_np).cuda(rpn_rois.get_device())

            labels_ = gt_classes[:, 1][gt_assignment[keep_inds]]
            bbox_target_ = gt_boxes[gt_assignment[keep_inds]]

            # Clamp labels_ for the background RoIs to 0
            if num_fg < labels_.size(0):
                labels_[num_fg:] = 0

            rois_ = torch.cat((
                im_sorted[t_start:t_end, None][keep_inds].float(),
                all_boxes[t_start:t_end][keep_inds],
            ), 1)

            labels.append(labels_)
            rois.append(rois_)
            bbox_targets.append(bbox_target_)

        rois = torch.cat(rois, 0)
        labels = torch.cat(labels, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        return rois, labels, bbox_targets


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', required=True, help='config file')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    # parser.add_argument('--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument('--set', dest='set_cfgs', help='Set config keys, overwriting config in the cfg_file. See lib/core/config.py for all options',
                        default=[], nargs='*')

    return parser.parse_args()


def main():
    # python tools/test_net.py --dataset coco2017 --cfg config/baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml --load_ckpt {path/to/your/checkpoint}
    print(sys.argv, len(sys.argv))
    sys.argv[1:] = ['--cfg', 'config/baselines/e2e_mask_rcnn_R-50-C4_2x.yaml',
                    '--load_ckpt', 'data/pretrained_model/e2e_mask_rcnn_R-50-C4_2x.pkl']
    print(sys.argv, len(sys.argv))
    print(parse_args())
    # bm = BaseModel()


if __name__ == '__main__':
    main()
