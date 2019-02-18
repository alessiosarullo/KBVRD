import logging

import numpy as np
import torch
from torch import nn

from lib.pydetectron_integration.box_utils import bbox_transform, clip_tiled_boxes
from ..core.config import cfg
from ..model.nms.nms_gpu import nms_gpu

logger = logging.getLogger(__name__)


class GenerateProposalsOp(nn.Module):
    def __init__(self, anchors, spatial_scale):
        super().__init__()
        self._anchors = anchors
        self._anchors_t = torch.tensor(self._anchors[None, :, :])
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = 1. / spatial_scale

    def forward(self, rpn_cls_prob, rpn_bbox_pred, im_info):
        """Op for generating RPN porposals.

        blobs_in:
          - 'rpn_cls_probs': 4D tensor of shape (N, A, H, W), where N is the
            number of minibatch images, A is the number of anchors per
            locations, and (H, W) is the spatial size of the prediction grid.
            Each value represents a "probability of object" rating in [0, 1].
          - 'rpn_bbox_pred': 4D tensor of shape (N, 4 * A, H, W) of predicted
            deltas for transformation anchor boxes into RPN proposals.
          - 'im_info': 2D tensor of shape (N, 3) where the three columns encode
            the input image's [height, width, scale]. Height and width are
            for the input to the network, not the original image; scale is the
            scale factor used to scale the original image to the network input
            size.

        blobs_out:
          - 'rpn_rois': 2D tensor of shape (R, 5), for R RPN proposals where the
            five columns encode [batch ind, x1, y1, x2, y2]. The boxes are
            w.r.t. the network input, which is a *scaled* version of the
            original image; these proposals must be scaled by 1 / scale (where
            scale comes from im_info; see above) to transform it back to the
            original input image coordinate system.
          - 'rpn_roi_probs': 1D tensor of objectness probability scores
            (extracted from rpn_cls_probs; see above).
        """
        # 1. for each location i in a (H, W) grid:
        #      generate A anchor boxes centered on cell i
        #      apply predicted bbox deltas to each of the A anchors at cell i
        # 2. clip predicted boxes to image
        # 3. remove predicted boxes with either height or width < threshold
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take the top pre_nms_topN proposals before NMS
        # 6. apply NMS with a loose threshold (0.7) to the remaining proposals
        # 7. take after_nms_topN proposals after NMS
        # 8. return the top proposals

        scores = rpn_cls_prob
        bbox_deltas = rpn_bbox_pred
        im_info = im_info.to(bbox_deltas)
        anchors = self._anchors_t

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]
        # Enumerate all shifted positions on the (H, W) grid
        shift_x = torch.arange(0, width) * self._feat_stride
        shift_y = torch.arange(0, height) * self._feat_stride
        shift_x, shift_y = torch.meshgrid([shift_x, shift_y])
        # Convert to (K, 4), K=H*W, where the columns are (dx, dy, dx, dy)
        # shift pointing to each grid location
        shifts = torch.stack([shift_x.reshape(-1),
                              shift_y.reshape(-1),
                              shift_x.reshape(-1),
                              shift_y.reshape(-1)], dim=1).double()

        # Broadcast anchors over shifts to enumerate all anchors at all positions
        # in the (H, W) grid:
        #   - add A anchors of shape (1, A, 4) to
        #   - K shifts of shape (K, 1, 4) to get
        #   - all shifted anchors of shape (K, A, 4)
        #   - reshape to (K*A, 4) shifted anchors
        num_images = scores.shape[0]
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = anchors + shifts.view(shifts.shape[0], 1, shifts.shape[1])
        all_anchors = all_anchors.to(bbox_deltas).view((K * A, 4))

        rois, roi_probs = [], []
        for im_i in range(num_images):
            im_i_boxes, im_i_probs = self.proposals_for_one_image(im_info[im_i, :], all_anchors, bbox_deltas[im_i, :, :, :], scores[im_i, :, :, :])
            batch_inds = torch.full_like(im_i_probs, fill_value=im_i)
            im_i_rois = torch.cat([batch_inds.view(-1, 1), im_i_boxes], dim=1)
            rois.append(im_i_rois)
            roi_probs.append(im_i_probs)
        rois = torch.cat(rois, dim=0)
        roi_probs = torch.cat(roi_probs, dim=0)
        return rois, roi_probs

    def proposals_for_one_image(self, im_info, all_anchors, bbox_deltas, scores):
        # Get mode-dependent configuration
        cfg_key = 'TRAIN' if self.training else 'TEST'
        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
        min_size = cfg[cfg_key].RPN_MIN_SIZE
        # print('generate_proposals:', pre_nms_topN, post_nms_topN, nms_thresh, min_size)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #   - bbox deltas will be (4 * A, H, W) format from conv output
        #   - transpose to (H, W, 4 * A)
        #   - reshape to (H * W * A, 4) where rows are ordered by (H, W, A)
        #     in slowest to fastest order to match the enumerated anchors
        bbox_deltas = bbox_deltas.permute(1, 2, 0).reshape(-1, 4)

        # Same story for the scores:
        #   - scores are (A, H, W) format from conv output
        #   - transpose to (H, W, A)
        #   - reshape to (H * W * A, 1) where rows are ordered by (H, W, A)
        #     to match the order of anchors and bbox_deltas
        scores = scores.permute(1, 2, 0).reshape(-1, 1)
        # print('pre_nms:', bbox_deltas.shape, scores.shape)

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        if pre_nms_topN <= 0 or pre_nms_topN >= len(scores):
            order = torch.sort(scores.view(-1), descending=True)[1]
        else:
            # Avoid sorting possibly large arrays; First partition to get top K unsorted and then sort just those (~20x faster for 200k scores)
            # FIXME numpy conversion
            scores_np = scores.cpu().numpy()
            inds = np.argpartition(-scores_np.squeeze(), pre_nms_topN)[:pre_nms_topN]
            order = np.argsort(-scores_np[inds].squeeze())
            order = inds[order]
            order = torch.from_numpy(order)
        bbox_deltas = bbox_deltas[order, :]
        all_anchors = all_anchors[order, :]
        scores = scores[order]

        # Transform anchors into proposals via bbox transformations
        proposals = bbox_transform(all_anchors, bbox_deltas, (1.0, 1.0, 1.0, 1.0), cfg.BBOX_XFORM_CLIP)

        # 2. clip proposals to image (may result in proposals with zero area
        # that will be removed in the next step)
        proposals = clip_tiled_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < min_size
        keep = _filter_boxes(proposals, min_size, im_info)
        proposals = proposals[keep, :]
        scores = scores[keep]
        # print('pre_nms:', proposals.shape, scores.shape)

        # 6. apply loose nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        if nms_thresh > 0:
            # keep = scores.new_tensor(scores.size(0), dtype=torch.int32)
            # num_out = nms.nms_apply(keep, proposals, nms_thresh)
            # keep = keep[:min(post_nms_topN, num_out)].long().to(scores.get_device())
            keep = nms_gpu(torch.cat([proposals, scores], dim=1), nms_thresh).long()
            proposals = proposals[keep, :]
            scores = scores[keep]
        # print('final proposals:', proposals.shape, scores.shape)
        print(proposals.shape, scores.shape)
        return proposals, scores


def _filter_boxes(boxes, min_size, im_info):
    """Only keep boxes with both sides >= min_size and center within the image."""

    # Scale min_size to match image scale
    min_size *= im_info[2]
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    x_ctr = boxes[:, 0] + ws / 2.
    y_ctr = boxes[:, 1] + hs / 2.
    keep = torch.nonzero((ws >= min_size) & (hs >= min_size) & (x_ctr < im_info[1]) & (y_ctr < im_info[0])).view(-1)
    return keep
