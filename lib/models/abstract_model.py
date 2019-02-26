import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn

from config import cfg
from lib.bbox_utils import iou_match_in_img, compute_ious, get_union_boxes
from lib.containers import Minibatch, Prediction
from lib.dataset.hicodet import HicoDetSplit
from .context import SpatialContext, ObjectContext
from .mask_rcnn import MaskRCNN


class AbstractModel(nn.Module):
    def __init__(self, dataset: HicoDetSplit, **kwargs):
        super().__init__()

        self.dataset = dataset
        self.mask_rcnn = MaskRCNN()

        # Derived
        self.mask_rcnn_vis_feat_dim = self.mask_rcnn.output_feat_dim

        # Branches
        self.spatial_context_branch = SpatialContext(input_dim=2 * (self.mask_rcnn.mask_resolution ** 2))
        self.obj_branch = ObjectContext(input_dim=self.mask_rcnn_vis_feat_dim +
                                                  self.dataset.num_object_classes +
                                                  self.spatial_context_branch.output_dim)
        self.hoi_branch = self._get_hoi_branch()

        self.obj_output_fc = nn.Linear(self.obj_branch.output_feat_dim, self.dataset.num_object_classes)
        self.hoi_output_fc = nn.Linear(self.hoi_branch.output_dim, self.dataset.num_predicates)

    def _get_hoi_branch(self):
        raise NotImplementedError()

    def get_losses(self, x, **kwargs):
        obj_output, rel_output, box_labels, rel_labels = self(x)
        class_loss = nn.functional.cross_entropy(obj_output, box_labels)
        rel_loss = nn.functional.cross_entropy(rel_output, rel_labels)
        return {'class_loss': class_loss, 'rel_loss': rel_loss}

    def forward(self, x, **kwargs):
        # TODO docs

        with torch.set_grad_enabled(self.training):
            tmp_output = self.first_step(x)
            boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos = tmp_output[:5]
            # `hoi_infos` is an R x 3 NumPy array where each column is [image ID, subject index, object index].

            if self.training:
                assert hoi_infos is not None
                obj_output, rel_output = self._forward(boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos)
                box_labels, rel_labels = tmp_output[5:]
                return obj_output, rel_output, box_labels, rel_labels
            else:
                if hoi_infos is not None:
                    assert boxes_ext is not None
                    obj_output, rel_output = self._forward(boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos)
                    obj_prob = nn.functional.softmax(obj_output, dim=1).cpu().numpy()
                    hoi_probs = nn.functional.softmax(rel_output, dim=1).cpu().numpy()
                    boxes_ext = boxes_ext.cpu().numpy()
                    obj_im_inds = boxes_ext[:, 0]
                    obj_boxes = boxes_ext[:, 1:5]
                    hoi_img_inds = hoi_infos[:, 0]
                    ho_pairs = hoi_infos[:, 1:]
                else:
                    hoi_probs = ho_pairs = hoi_img_inds = None
                    if boxes_ext is None:
                        obj_im_inds = obj_boxes = obj_prob = None
                    else:
                        boxes_ext = boxes_ext.cpu().numpy()
                        obj_im_inds = boxes_ext[:, 0]
                        obj_boxes = boxes_ext[:, 1:5]
                        obj_prob = boxes_ext[:, 5:]
                return Prediction(obj_im_inds=obj_im_inds,
                                  obj_boxes=obj_boxes,
                                  obj_scores=obj_prob,
                                  hoi_img_inds=hoi_img_inds,
                                  ho_pairs=ho_pairs,
                                  hoi_scores=hoi_probs)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos):
        # TODO docs

        # Compute quantities used later
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        hoi_im_ids = hoi_infos[:, 0]
        sub_inds = hoi_infos[:, 1]
        obj_inds = hoi_infos[:, 2]
        im_ids = torch.unique(hoi_im_ids, sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        spatial_ctx = self.spatial_context_branch([masks, im_ids, hoi_im_ids, sub_inds, obj_inds])
        obj_ctx, objs_feats = self.obj_branch(boxes_ext, box_feats, spatial_ctx, im_ids, box_im_ids)
        hoi_feats = self.hoi_branch([union_boxes_feats, box_feats, spatial_ctx, obj_ctx, im_ids, hoi_im_ids, sub_inds, obj_inds])

        obj_output = self.obj_output_fc(objs_feats)
        hoi_output = self.hoi_output_fc(hoi_feats)
        return obj_output, hoi_output

    def first_step(self, batch: Minibatch):  # FIXME change name
        """
        :param batch:
        :param kwargs:
        :return:
        """
        # TODO docs

        boxes_ext_np, masks, feat_map, box_feats = self.mask_rcnn(batch)
        assert boxes_ext_np.shape[0] == box_feats.shape[0] == masks.shape[0]
        # `boxes_ext_np` is Bx(1+4+C) where each row is [im_id, bbox_coord, class_scores]. Classes are COCO ones.

        boxes_ext_np, masks, box_feats = self.filter_and_map_to_hico(boxes_ext_np, masks, box_feats)

        if self.training:
            boxes_ext_np, box_labels, box_feats, masks = self.box_gt_assignment(batch, boxes_ext_np, box_feats, masks, feat_map)
            rel_im_ids, ho_pairs, rel_labels = self.rel_gt_assignments(batch, boxes_ext_np)
            assert rel_im_ids.shape[0] == rel_labels.shape[0] == ho_pairs.shape[0]
            assert box_labels.shape[0] == boxes_ext_np.shape[0] == box_feats.shape[0] == masks.shape[0]
        else:
            if cfg.program.predcls:
                # This is inefficient because Mask-RCNN has already been called at this point/features have been loaded, but on irrelevant boxes.

                boxes_with_im_id = np.concatenate([batch.gt_box_im_ids[:, None], batch.gt_boxes], axis=1)

                box_feats = self.mask_rcnn.get_rois_feats(fmap=feat_map, rois=boxes_with_im_id)
                masks = self.mask_rcnn.get_masks(img_infos=batch.img_infos,
                                                 fmap=feat_map,
                                                 boxes=batch.gt_boxes,
                                                 box_im_ids=batch.gt_box_im_ids,
                                                 box_classes=batch.gt_obj_classes)
                labels_onehot = np.zeros((boxes_with_im_id.shape[0], self.dataset.num_object_classes))
                labels_onehot[np.arange(boxes_with_im_id.shape[0]), batch.gt_obj_classes] = 1
                boxes_ext_np = np.concatenate([boxes_with_im_id, labels_onehot], axis=1)
            else:
                if boxes_ext_np.shape[0] == 0:
                    return None, None, None, None, None
            rel_im_ids, ho_pairs = self.get_all_pairs(boxes_ext_np)
        boxes_ext = torch.tensor(boxes_ext_np, device=masks.device, dtype=torch.float32)

        if rel_im_ids.size == 0:
            assert not self.training and not cfg.program.predcls
            return boxes_ext, box_feats, masks, None, None
        assert ho_pairs.shape[0] > 0

        # Note that box indices in `ho_pairs` are over all boxes, NOT relative to each specific image
        rel_union_boxes = get_union_boxes(boxes_ext_np[:, 1:5], ho_pairs)
        union_boxes_feats = self.mask_rcnn.get_rois_feats(fmap=feat_map, rois=rel_union_boxes)
        assert rel_im_ids.shape[0] == union_boxes_feats.shape[0]

        rel_infos = np.concatenate([rel_im_ids[:, None], ho_pairs], axis=1).astype(np.int, copy=False)
        if self.training:
            box_labels = torch.tensor(box_labels, device=masks.device)
            rel_labels = torch.tensor(rel_labels, device=masks.device)
            return boxes_ext, box_feats, masks, union_boxes_feats, rel_infos, box_labels, rel_labels
        else:
            return boxes_ext, box_feats, masks, union_boxes_feats, rel_infos

    def filter_and_map_to_hico(self, boxes_ext: np.ndarray, masks: torch.Tensor, box_feats: torch.Tensor):
        # Keep foreground object only
        class_scores = boxes_ext[:, 5:]
        classes = np.argmax(class_scores, axis=1)
        fg_inds = (classes > 0)
        assert fg_inds.shape[0] == boxes_ext.shape[0] == masks.shape[0]
        fg_inds = np.flatnonzero(fg_inds)  # this is needed for torch tensors
        boxes_ext_fg, masks, box_feats = boxes_ext[fg_inds, :], masks[fg_inds, :], box_feats[fg_inds, :]

        # Map from COCO classes to HICO ones
        coco_to_hico_mapping = np.array(self.dataset.hicodet.map_coco_classes_to_hico(), dtype=np.int)
        boxes_ext_hico = boxes_ext_fg[:, np.concatenate([np.arange(5), 5 + coco_to_hico_mapping])]  # convert to Hico classes by swapping columns

        return boxes_ext_hico, masks, box_feats

    def get_all_pairs(self, boxes_ext, box_classes=None):
        # FIXME there is a significant overlap between this method and rel_gt_assignment. Merge.
        box_classes = box_classes or np.argmax(boxes_ext[:, 5:], axis=1)
        person_box_inds = (box_classes == self.dataset.hicodet.person_class)

        person_boxes_ext = boxes_ext[person_box_inds, :]
        if person_boxes_ext.shape[0] == 0:
            return np.empty([0, ]), np.empty([0, 2])

        if self.filter_rels_of_non_overlapping_boxes:
            _, pred_box_ious = iou_match_in_img(person_boxes_ext[:, :5], boxes_ext[:, :5])
            possible_rels_mat = (0 < pred_box_ious) & (pred_box_ious < 1)
            subjs, objs = np.where(possible_rels_mat)
            subjs = np.flatnonzero(person_box_inds)[subjs]
        else:
            block_img_mat = (boxes_ext[:, 0][:, None] == boxes_ext[:, 0][None, :])
            assert block_img_mat.shape[0] == block_img_mat.shape[1]
            possible_rels_mat = block_img_mat - np.eye(block_img_mat.shape[0])
            possible_rels_mat = possible_rels_mat[person_box_inds, :]
            subjs, objs = np.where(possible_rels_mat)
        rel_im_ids = boxes_ext[subjs, 0]
        assert np.all(rel_im_ids == boxes_ext[objs, 0])
        sub_obj_pairs = np.stack([subjs, objs], axis=1)  # this is over the original boxes, not person ones
        return rel_im_ids, sub_obj_pairs

    def box_gt_assignment(self, batch: Minibatch, boxes_ext, box_feats, masks, feat_map, gt_iou_thr=0.5):
        gt_boxes_with_imid = np.concatenate([batch.gt_box_im_ids[:, None], batch.gt_boxes], axis=1)

        gt_idx_per_pred_box, pred_gt_box_ious = iou_match_in_img(boxes_ext[:, :5], gt_boxes_with_imid)
        obj_labels = batch.gt_obj_classes[gt_idx_per_pred_box]
        gt_match = np.flatnonzero(np.any(pred_gt_box_ious >= gt_iou_thr, axis=1))
        boxes_ext = boxes_ext[gt_match, :]
        obj_labels = obj_labels[gt_match]
        box_feats = box_feats[gt_match, :]
        masks = masks[gt_match, :]

        unmatched_gt_boxes_inds = np.flatnonzero(np.all(pred_gt_box_ious < gt_iou_thr, axis=0))
        unmatched_gt_obj_labels = batch.gt_obj_classes[unmatched_gt_boxes_inds]
        unmatched_gt_labels_onehot = np.zeros((unmatched_gt_boxes_inds.size, self.dataset.num_object_classes))
        unmatched_gt_labels_onehot[np.arange(unmatched_gt_boxes_inds.size), unmatched_gt_obj_labels] = 1
        unmatched_gt_boxes_ext = np.concatenate([gt_boxes_with_imid[unmatched_gt_boxes_inds, :], unmatched_gt_labels_onehot], axis=1)
        unmatched_gt_boxes = unmatched_gt_boxes_ext[:, 1:5]
        unmatched_gt_box_im_inds = unmatched_gt_boxes_ext[:, 0]

        unmatched_gt_boxes_feats = self.mask_rcnn.get_rois_feats(fmap=feat_map, rois=unmatched_gt_boxes)
        unmatched_gt_boxes_masks = self.mask_rcnn.get_masks(img_infos=batch.img_infos,
                                                            fmap=feat_map,
                                                            boxes=unmatched_gt_boxes,
                                                            box_im_ids=unmatched_gt_box_im_inds,
                                                            box_classes=unmatched_gt_obj_labels)

        boxes_ext = np.concatenate([boxes_ext, unmatched_gt_boxes_ext], axis=0)
        obj_labels = np.concatenate([obj_labels, unmatched_gt_obj_labels], axis=0)
        box_feats = torch.cat([box_feats, unmatched_gt_boxes_feats], dim=0)
        masks = torch.cat([masks, unmatched_gt_boxes_masks], dim=0)
        return boxes_ext, obj_labels, box_feats, masks

    def rel_gt_assignments(self, batch: Minibatch, boxes_ext_np, num_sample_per_gt=4, filter_non_overlap=False, fg_rels_per_image=16):
        gt_boxes, gt_box_im_ids, gt_obj_classes = batch.gt_boxes, batch.gt_box_im_ids, batch.gt_obj_classes
        gt_inters, gt_inters_im_ids = batch.gt_hois, batch.gt_hoi_im_ids
        predict_box_im_ids = boxes_ext_np[:, 0]
        predict_boxes = boxes_ext_np[:, 1:5]
        predict_box_classes = np.argmax(boxes_ext_np[:, 5:], axis=1)

        rel_infos = []
        num_box_seen = 0
        for im_id in np.unique(gt_box_im_ids):
            predict_box_im_ids_i = (predict_box_im_ids == im_id)
            gt_box_im_ids_i = (gt_box_im_ids == im_id)
            assert np.any(predict_box_im_ids_i)

            gt_boxes_i = gt_boxes[gt_box_im_ids_i]
            gt_obj_classes_i = gt_obj_classes[gt_box_im_ids_i]
            gt_rels_i = gt_inters[gt_inters_im_ids == im_id]

            predict_boxes_i = predict_boxes[predict_box_im_ids_i]
            predict_box_labels_i = predict_box_classes[predict_box_im_ids_i]
            predict_human_boxes_i = (predict_box_labels_i == self.dataset.hicodet.person_class)

            iou_predict_to_gt_i = compute_ious(predict_boxes_i, gt_boxes_i)
            predict_gt_match = (predict_box_labels_i[:, None] == gt_obj_classes_i[None, :]) & (iou_predict_to_gt_i >= 0.5)  # FIXME magic constant

            human_subject_possibilities = np.zeros((predict_boxes_i.shape[0], predict_boxes_i.shape[0]), dtype=bool)
            human_subject_possibilities[predict_human_boxes_i, :] = True
            human_subject_possibilities[np.arange(predict_boxes_i.shape[0]), np.arange(predict_boxes_i.shape[0])] = False
            rel_possibilities = human_subject_possibilities
            if filter_non_overlap:
                # Limit to IOUs that overlap, but are not the exact same box
                iou_predict_boxes_i = compute_ious(predict_boxes_i, predict_boxes_i)
                predict_boxes_intersect = (0 < iou_predict_boxes_i) & (iou_predict_boxes_i < 1)
                rel_possibilities = rel_possibilities & predict_boxes_intersect

            # Sample the GT relationships.
            fg_rels = []
            p_size = []
            for i, (from_gt_ind, rel_id, to_gt_ind) in enumerate(gt_rels_i):
                fg_rels_i = []
                fg_scores_i = []

                for from_predict_ind in np.flatnonzero(predict_gt_match[:, from_gt_ind]):
                    for to_predict_ind in np.flatnonzero(predict_gt_match[:, to_gt_ind]):
                        if from_predict_ind != to_predict_ind:
                            fg_rels_i.append((from_predict_ind, to_predict_ind, rel_id))
                            fg_scores_i.append((iou_predict_to_gt_i[from_predict_ind, from_gt_ind] * iou_predict_to_gt_i[to_predict_ind, to_gt_ind]))
                            rel_possibilities[from_predict_ind, to_predict_ind] = False
                if len(fg_rels_i) == 0:
                    continue
                p = np.array(fg_scores_i)
                p = p / p.sum()
                p_size.append(p.shape[0])
                num_to_add = min(p.shape[0], num_sample_per_gt)
                for rel_to_add in npr.choice(p.shape[0], p=p, size=num_to_add, replace=False):
                    fg_rels.append(fg_rels_i[rel_to_add])

            fg_rels = np.array(fg_rels, dtype=np.int64)
            if fg_rels.size > 0 and fg_rels.shape[0] > fg_rels_per_image:
                fg_rels = fg_rels[npr.choice(fg_rels.shape[0], size=fg_rels_per_image, replace=False)]
            elif fg_rels.size == 0:
                fg_rels = np.zeros((0, 3), dtype=np.int64)

            bg_rels = np.column_stack(np.where(rel_possibilities))
            bg_rels = np.column_stack((bg_rels, np.zeros(bg_rels.shape[0], dtype=np.int64)))

            num_bg_rel = min(64 - fg_rels.shape[0], bg_rels.shape[0])  # FIXME magic constant
            if bg_rels.size > 0:
                bg_rels = bg_rels[np.random.choice(bg_rels.shape[0], size=num_bg_rel, replace=False)]
            else:
                bg_rels = np.zeros((0, 3), dtype=np.int64)

            if fg_rels.size == 0 and bg_rels.size == 0:
                # Just put something here
                bg_rels = np.array([[0, 0, 0]], dtype=np.int64)

            # print("GTR {} -> AR {} vs {}".format(gt_rels.shape, fg_rels.shape, bg_rels.shape))
            all_rels_i = np.concatenate((fg_rels, bg_rels), axis=0)
            all_rels_i[:, :2] += num_box_seen

            all_rels_i = all_rels_i[np.lexsort((all_rels_i[:, 1], all_rels_i[:, 0]))]

            rel_infos.append(np.column_stack([np.full(all_rels_i.shape[0], fill_value=im_id, dtype=np.int64),
                                              all_rels_i]))

            num_box_seen += predict_boxes_i.shape[0]
        rel_infos = np.concatenate(rel_infos, axis=0)
        rel_im_ids = rel_infos[:, 0]
        sub_obj_pairs = rel_infos[:, 1:3]  # [sub_ind, obj_ind]
        rel_preds = rel_infos[:, 3]  # [pred]
        return rel_im_ids, sub_obj_pairs, rel_preds
