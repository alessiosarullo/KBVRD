import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

from analysis.utils import postprocess_for_visualisation
from config import cfg
from lib.bbox_utils import compute_ious, get_union_boxes
from lib.dataset.hicodet import HicoDetInstanceSplit, Splits
from lib.dataset.utils import Minibatch


# noinspection PyCallingNonCallable
class VisualModule(nn.Module):
    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__()
        self.gt_iou_thr = 0.5  # FIXME? params
        self.box_proposal_thr = cfg.model.proposal_thr
        self.dataset = dataset
        self._precomputed = self.dataset.has_precomputed

        if not self._precomputed:
            from lib.models.mask_rcnn import MaskRCNN
            self.mask_rcnn = MaskRCNN()
            self.mask_resolution = self.mask_rcnn.mask_resolution
            self.vis_feat_dim = self.mask_rcnn.output_feat_dim
        else:
            self.mask_resolution = cfg.model.mask_resolution
            self.vis_feat_dim = self.dataset.precomputed_visual_feat_dim

    def forward(self, batch: Minibatch, inference, **kwargs):
        # TODO docs
        # `ho_infos` is an R x 3 NumPy array where each column is [image ID, subject index, object index].
        # FIXME the returns here are horrible

        if self._precomputed:
            boxes_ext_np = batch.pc_boxes_ext
            ho_infos = batch.pc_ho_infos

            if self.box_proposal_thr > 0:
                valid_box_inds = boxes_ext_np[:, 5:].max(axis=1) > self.box_proposal_thr
                boxes_ext_np = boxes_ext_np[valid_box_inds, :]
                uncertain_boxes_set = set((~valid_box_inds).tolist())

                valid_hoi_inds = []
                for i, (im, h, o) in enumerate(ho_infos):
                    if h not in uncertain_boxes_set and o not in uncertain_boxes_set:
                        valid_hoi_inds.append(i)
                valid_hoi_inds = np.array(valid_hoi_inds)
                ho_infos = ho_infos[valid_hoi_inds, :]
                if ho_infos.shape[0] == 0 and not inference:
                    return None, None, None, None, None, None, None, None, None

            if inference and not cfg.program.predcls and boxes_ext_np.shape[0] == 0:
                return None, None, None, None, None, None, None, None, None

            b_pc_box_feats = batch.pc_box_feats
            b_pc_masks = batch.pc_masks
            b_pc_box_labels = batch.pc_box_labels
            b_pc_action_labels = batch.pc_action_labels
            hoi_union_boxes = batch.pc_ho_union_boxes
            b_pc_ho_union_feats = batch.pc_ho_union_feats
            if self.box_proposal_thr > 0:
                b_pc_box_feats = b_pc_box_feats[valid_box_inds, :]
                b_pc_masks = b_pc_masks[valid_box_inds, :]
                b_pc_box_labels = b_pc_box_labels[valid_box_inds, :]
                b_pc_action_labels = b_pc_action_labels[valid_hoi_inds, :]
                hoi_union_boxes = hoi_union_boxes[valid_hoi_inds, :]
                b_pc_ho_union_feats = b_pc_ho_union_feats[valid_hoi_inds, :]

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            boxes_ext = torch.tensor(boxes_ext_np, device=device)
            box_feats = torch.tensor(b_pc_box_feats, device=device)
            masks = torch.tensor(b_pc_masks, device=device)

            if ho_infos.shape[0] == 0:
                assert inference and not cfg.program.predcls
                return boxes_ext, box_feats, masks, None, None, None, None, None, None
            ho_infos = ho_infos.astype(np.int, copy=False)

            if b_pc_box_labels is None:
                assert b_pc_action_labels is None
                box_labels = action_labels = hoi_labels = None
            else:
                box_labels_np = b_pc_box_labels
                action_labels_np = b_pc_action_labels
                hoi_labels_np = self.action_labels_to_hoi_labels(box_labels_np, action_labels_np, ho_infos)

                box_labels = torch.tensor(box_labels_np, device=device)
                action_labels = torch.tensor(action_labels_np, device=device)
                hoi_labels = torch.tensor(hoi_labels_np, device=device)

            # Note that box indices in `ho_infos` are over all boxes, NOT relative to each specific image
            hoi_union_boxes_feats = torch.tensor(b_pc_ho_union_feats, device=device)
            return boxes_ext, box_feats, masks, hoi_union_boxes, hoi_union_boxes_feats, ho_infos, box_labels, action_labels, hoi_labels
        else:
            with torch.set_grad_enabled(self.training):
                boxes_ext_np, feat_map = self.mask_rcnn(batch)
                # `boxes_ext_np` is Bx(1+4+C) where each row is [im_id, bbox_coord, class_scores]. Classes are COCO ones.
                if inference and not cfg.program.predcls and boxes_ext_np.shape[0] == 0:
                    return None, None, None, None, None, None, None, None, None

                boxes_ext_np, box_feats, masks, ho_infos, box_labels, action_labels, hoi_labels = self.process_boxes(batch, inference, feat_map,
                                                                                                                     boxes_ext_np)
                boxes_ext = torch.tensor(boxes_ext_np, device=feat_map.device, dtype=torch.float32)

                if ho_infos.shape[0] == 0:
                    assert inference and not cfg.program.predcls
                    return boxes_ext, box_feats, masks, None, None, None, None, None, None
                ho_infos = ho_infos.astype(np.int, copy=False)

                # Note that box indices in `ho_infos` are over all boxes, NOT relative to each specific image
                hoi_union_boxes = get_union_boxes(boxes_ext_np[:, 1:5], ho_infos[:, 1:])
                hoi_union_boxes_feats = self.mask_rcnn.get_rois_feats(fmap=feat_map, rois=np.concatenate([ho_infos[:, :1], hoi_union_boxes], axis=1))
                assert ho_infos.shape[0] == hoi_union_boxes_feats.shape[0]
                return boxes_ext, box_feats, masks, hoi_union_boxes, hoi_union_boxes_feats, ho_infos, box_labels, action_labels, hoi_labels

    def action_labels_to_hoi_labels(self, box_labels, action_labels, ho_infos):
        # Requires everything Numpy
        box_labels_per_pair_np = box_labels[ho_infos[:, 2]].astype(np.int)
        hoi_labels_np = np.zeros((action_labels.shape[0], self.dataset.num_interactions), dtype=np.float32)
        for i in range(action_labels.shape[0]):
            a_inds = np.flatnonzero(action_labels[i, :])
            inter_inds = self.dataset.op_pair_to_interaction[np.full_like(a_inds, fill_value=box_labels_per_pair_np[i]), a_inds]
            assert np.all(inter_inds >= 0)
            hoi_labels_np[i, inter_inds] = 1
        return hoi_labels_np

    def process_boxes(self, batch, predict, feat_map, boxes_ext_np):
        if not predict:
            # Map from COCO classes to HICO ones by swapping columns
            boxes_ext_np = boxes_ext_np[:, np.concatenate([np.arange(5), 5 + self.dataset.hico_to_coco_mapping])]

            # FIXME match with GT boxes is done twice
            boxes_ext_np, box_classes = self.box_gt_assignment(batch, boxes_ext_np)
            ho_infos, action_labels = self.hoi_gt_assignments(batch, boxes_ext_np, box_classes)
            hoi_labels_np = self.action_labels_to_hoi_labels(box_classes, action_labels, ho_infos)

            box_labels = torch.tensor(box_classes, device=feat_map.device)
            action_labels = torch.tensor(action_labels, device=feat_map.device)
            hoi_labels = torch.tensor(hoi_labels_np, device=feat_map.device)
            assert ho_infos.shape[0] == action_labels.shape[0] == hoi_labels.shape[0]
            assert box_labels.shape[0] == boxes_ext_np.shape[0]
        else:
            if cfg.program.predcls:
                box_classes = batch.gt_obj_classes
                boxes_ext_np = np.concatenate([batch.gt_box_im_ids[:, None], batch.gt_boxes, self.one_hot_obj_labels(box_classes)], axis=1)
            else:
                # Keep foreground object only, then map from COCO classes to HICO ones by swapping columns
                coco_box_classes = np.argmax(boxes_ext_np[:, 5:], axis=1)
                if np.all(coco_box_classes == 0):  # all boxes would be discarded. Keep just the first one.
                    boxes_ext_np = boxes_ext_np[:1, :]
                else:
                    boxes_ext_np = boxes_ext_np[coco_box_classes > 0, :]
                assert boxes_ext_np.shape[0] > 0
                boxes_ext_np = boxes_ext_np[:, np.concatenate([np.arange(5), 5 + self.dataset.hico_to_coco_mapping])]
                box_classes = np.argmax(boxes_ext_np[:, 5:], axis=1)

            # Sort by image
            inds = np.argsort(boxes_ext_np[:, 0]).astype(np.int64, copy=False)
            boxes_ext_np = boxes_ext_np[inds]
            box_classes = box_classes[inds]

            ho_infos = self.get_all_pairs(boxes_ext_np, box_classes)
            box_labels = action_labels = hoi_labels = None

        masks = self.mask_rcnn.get_masks(fmap=feat_map, rois=boxes_ext_np[:, :5], box_classes=self.dataset.hico_to_coco_mapping[box_classes])
        box_feats = self.mask_rcnn.get_rois_feats(fmap=feat_map, rois=boxes_ext_np[:, :5])
        assert boxes_ext_np.shape[0] == box_feats.shape[0] == masks.shape[0]

        return boxes_ext_np, box_feats, masks, ho_infos, box_labels, action_labels, hoi_labels

    def box_gt_assignment(self, batch: Minibatch, boxes_ext):
        gt_rois = np.concatenate([batch.gt_box_im_ids[:, None], batch.gt_boxes], axis=1)

        pred_gt_box_ious = self.iou_match_in_img(boxes_ext[:, :5], gt_rois)
        pred_gt_best_match = np.argmax(pred_gt_box_ious, axis=1)  # type: np.ndarray
        box_labels = batch.gt_obj_classes[pred_gt_best_match]  # assign the best match

        has_overlap_with_gt = np.flatnonzero(np.any(pred_gt_box_ious >= self.gt_iou_thr, axis=1))  # filter if not good enough
        boxes_ext = boxes_ext[has_overlap_with_gt, :]
        box_labels = box_labels[has_overlap_with_gt]
        pred_gt_match = pred_gt_best_match[has_overlap_with_gt]

        unmatched_gt_boxes_inds = np.array(sorted(set(range(gt_rois.shape[0])) - set(pred_gt_match.tolist())))
        if unmatched_gt_boxes_inds.size > 0:
            unm_gt_box_labels = batch.gt_obj_classes[unmatched_gt_boxes_inds]
            unm_gt_boxes_ext = np.concatenate([gt_rois[unmatched_gt_boxes_inds, :], self.one_hot_obj_labels(unm_gt_box_labels)], axis=1)

            box_labels = np.concatenate([box_labels, unm_gt_box_labels], axis=0)
            boxes_ext = np.concatenate([boxes_ext, unm_gt_boxes_ext], axis=0)

        # Sort by image
        inds = np.argsort(boxes_ext[:, 0]).astype(np.int64, copy=False)
        boxes_ext = boxes_ext[inds]
        box_labels = box_labels[inds]

        return boxes_ext, box_labels

    def hoi_gt_assignments(self, batch: Minibatch, boxes_ext, box_labels, resample_bg=False):
        bg_ratio = cfg.opt.hoi_bg_ratio

        gt_boxes, gt_box_im_ids, gt_box_classes = batch.gt_boxes, batch.gt_box_im_ids, batch.gt_obj_classes
        gt_inters, gt_inters_im_ids = batch.gt_hois, batch.gt_hoi_im_ids
        predict_box_im_ids = boxes_ext[:, 0]
        predict_boxes = boxes_ext[:, 1:5]

        ho_infos_and_action_labels = []
        num_box_seen = 0
        for im_id in np.unique(gt_box_im_ids):
            # Get image values
            predict_box_im_ids_i = (predict_box_im_ids == im_id)
            gt_box_im_ids_i = (gt_box_im_ids == im_id)
            assert np.any(predict_box_im_ids_i)

            gt_boxes_i = gt_boxes[gt_box_im_ids_i]
            gt_box_classes_i = gt_box_classes[gt_box_im_ids_i]
            gt_rels_i = gt_inters[gt_inters_im_ids == im_id]

            predict_boxes_i = predict_boxes[predict_box_im_ids_i]
            predict_box_labels_i = box_labels[predict_box_im_ids_i]
            num_predict_boxes_i = predict_boxes_i.shape[0]

            # Find rel distribution
            iou_predict_to_gt_i = compute_ious(predict_boxes_i, gt_boxes_i)
            predict_gt_match_i = (predict_box_labels_i[:, None] == gt_box_classes_i[None, :]) & (iou_predict_to_gt_i >= self.gt_iou_thr)

            action_labels_i = np.zeros((num_predict_boxes_i, num_predict_boxes_i, self.dataset.num_predicates))
            for head_gt_ind, rel_id, tail_gt_ind in gt_rels_i:
                for head_predict_ind in np.flatnonzero(predict_gt_match_i[:, head_gt_ind]):
                    for tail_predict_ind in np.flatnonzero(predict_gt_match_i[:, tail_gt_ind]):
                        if head_predict_ind != tail_predict_ind:
                            action_labels_i[head_predict_ind, tail_predict_ind, rel_id] = 1.0

            ho_fg_mask = action_labels_i[:, :, 1:].any(axis=2)
            assert not np.any(action_labels_i[:, :, 0].astype(bool) & ho_fg_mask)  # it's either foreground or background
            ho_bg_mask = ~ho_fg_mask
            action_labels_i[:, :, 0] = ho_bg_mask.astype(np.float)

            # Filter irrelevant BG relationships (i.e., those where the subject is not human).
            non_human_box_inds_i = (predict_box_labels_i != self.dataset.human_class)
            ho_bg_mask[non_human_box_inds_i, :] = 0

            ho_fg_pairs_i = np.stack(np.where(ho_fg_mask), axis=1)
            ho_bg_pairs_i = np.stack(np.where(ho_bg_mask), axis=1)
            num_bg_to_sample = ho_fg_pairs_i.shape[0] * bg_ratio
            bg_inds = np.random.permutation(ho_bg_pairs_i.shape[0])[:num_bg_to_sample]
            if resample_bg and bg_inds.size < num_bg_to_sample:  # resample randomly to get to the chosen number
                bg_inds = np.concatenate([bg_inds, np.random.choice(bg_inds, size=num_bg_to_sample - bg_inds.size, replace=True)])
            ho_bg_pairs_i = ho_bg_pairs_i[bg_inds, :]

            ho_pairs_i = np.concatenate([ho_fg_pairs_i, ho_bg_pairs_i], axis=0)
            ho_infos_i = np.stack([np.full(ho_pairs_i.shape[0], fill_value=im_id),
                                   ho_pairs_i[:, 0] + num_box_seen,
                                   ho_pairs_i[:, 1] + num_box_seen], axis=1)
            if ho_infos_i.shape[0] == 0:  # since GT boxes are added to predicted ones during training this cannot be empty
                print(gt_boxes_i)
                print(predict_boxes_i)
                print(gt_box_classes_i)
                print(predict_box_labels_i)
                raise RuntimeError
            ho_infos_and_action_labels.append(np.concatenate([ho_infos_i, action_labels_i[ho_pairs_i[:, 0], ho_pairs_i[:, 1]]], axis=1))
            num_box_seen += num_predict_boxes_i

        ho_infos_and_action_labels = np.concatenate(ho_infos_and_action_labels, axis=0)
        ho_infos = ho_infos_and_action_labels[:, :3].astype(np.int, copy=False)  # [im_id, sub_ind, obj_ind]
        action_labels = ho_infos_and_action_labels[:, 3:].astype(np.float32, copy=False)  # [pred]
        return ho_infos, action_labels

    def get_all_pairs(self, boxes_ext, box_classes):
        human_box_inds = (box_classes == self.dataset.human_class)

        if human_box_inds.size == 0:
            return np.empty([0, 3])
        else:
            block_img_mat = (boxes_ext[:, 0][:, None] == boxes_ext[:, 0][None, :])
            assert block_img_mat.shape[0] == block_img_mat.shape[1]
            possible_rels_mat = block_img_mat - np.eye(block_img_mat.shape[0])
            possible_rels_mat[~human_box_inds, :] = 0  # only from human
            # possible_rels_mat[:, human_box_inds] = 0  # only to non-human
            hum_inds, obj_inds = np.where(possible_rels_mat)

            hoi_im_ids = boxes_ext[hum_inds, 0]
            assert np.all(hoi_im_ids == boxes_ext[obj_inds, 0])
            ho_infos = np.stack([hoi_im_ids, hum_inds, obj_inds], axis=1).astype(np.int, copy=False)  # box indices are over the original boxes
            return ho_infos

    def one_hot_obj_labels(self, labels: np.ndarray):
        labels_onehot = np.zeros((labels.shape[0], self.dataset.num_object_classes))
        labels_onehot[np.arange(labels_onehot.shape[0]), labels] = 1
        return labels_onehot

    @staticmethod
    def iou_match_in_img(boxes1, boxes2):
        box_im_ids1 = boxes1[:, 0]
        box_im_ids2 = boxes2[:, 0]
        ious = compute_ious(boxes1[:, 1:5], boxes2[:, 1:5])
        ious[box_im_ids1[:, None] != box_im_ids2[None, :]] = 0.0
        return ious

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


def main():
    from analysis.utils import vis_one_image

    np.set_printoptions(precision=3, suppress=True)

    sys.argv += ['--model', 'base', '--save_dir', 'fake']  # fake required arguments
    cfg.parse_args()
    output_dir = os.path.join('analysis', 'output', 'tmp', 'e2e' if cfg.program.recompute_visual else 'pre')
    os.makedirs(output_dir, exist_ok=True)

    seed = 3 if not cfg.program.randomize else np.random.randint(1_000_000_000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print('RNG seed:', seed)

    train_split = HicoDetInstanceSplit.get_split(split=Splits.TRAIN, flipping_prob=cfg.data.flip_prob)
    vm = VisualModule(dataset=train_split)
    vm.cuda()
    vm.eval()
    split = Splits.TRAIN
    if split == Splits.TRAIN:
        hds = train_split
    else:
        hds = HicoDetInstanceSplit.get_split(split=Splits.TEST, im_inds=[0, 1, 2, 3, 4])
    hdsl = hds.get_loader(batch_size=1, shuffle=False)

    for batch_i, batch in enumerate(hdsl):
        print('Batch', batch_i)
        batch = batch  # type: Minibatch
        assert len(batch.other_ex_data) == 1

        boxes_ext, box_feats, masks, union_boxes, union_boxes_feats, ho_infos, _, _ = vm(batch, mode_inference=split == Splits.TEST)

        im_fn = batch.other_ex_data[0]['fn']
        im = cv2.imread(os.path.join(hds.img_dir, im_fn))

        boxes_with_scores, box_classes, masks, union_boxes = postprocess_for_visualisation(boxes_ext, masks, union_boxes, batch.img_infos)

        print(union_boxes)
        print(ho_infos)
        print(np.concatenate([boxes_with_scores, box_classes[:, None]], axis=1))

        vis_one_image(
            im[:, :, [2, 1, 0]],  # BGR -> RGB for visualization
            boxes=boxes_with_scores,
            box_classes_scores=box_classes,
            object_names=hds.objects,
            masks=masks,
            union_boxes=union_boxes,
            output_file_path=os.path.join(output_dir, os.path.splitext(im_fn)[0]),
            box_alpha=0.3,
            show_class=True,
            box_thr=0.0,  # Lower this to see all the predictions (was 0.7 in the original code)
            ext='png'
        )


if __name__ == '__main__':
    main()
