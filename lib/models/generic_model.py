import numpy as np
import torch
import torch.nn as nn

from config import cfg
from lib.bbox_utils import iou_match_in_img, compute_ious, get_union_boxes
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.dataset.utils import Minibatch
from lib.models.abstract_model import AbstractModel
from lib.models.mask_rcnn import MaskRCNN
from lib.models.utils import Prediction


class GenericModel(AbstractModel):
    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.mask_rcnn = MaskRCNN()
        self.filter_rels_of_non_overlapping_boxes = False  # TODO? create config for this
        self.mask_rcnn_vis_feat_dim = self.mask_rcnn.output_feat_dim

    def get_losses(self, x, **kwargs):
        obj_output, hoi_output, box_labels, hoi_labels = self(x, predict=False, **kwargs)
        obj_loss = nn.functional.cross_entropy(obj_output, box_labels)
        hoi_loss = nn.functional.binary_cross_entropy_with_logits(hoi_output, hoi_labels)
        return {'object_loss': obj_loss, 'hoi_loss': hoi_loss}

    def forward(self, x, predict=True, **kwargs):
        # TODO docs

        with torch.set_grad_enabled(self.training):
            boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels, hoi_labels = self.detection_and_gt_assignment(x, predict)
            # `hoi_infos` is an R x 3 NumPy array where each column is [image ID, subject index, object index].

            if not predict:
                assert hoi_infos is not None and box_labels is not None and hoi_labels is not None
                obj_output, hoi_output = self._forward(boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels, hoi_labels)
                return obj_output, hoi_output, box_labels, hoi_labels
            else:
                if hoi_infos is not None:
                    assert boxes_ext is not None
                    obj_output, hoi_output = self._forward(boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels, hoi_labels)
                    if cfg.program.predcls:
                        obj_prob = None  # this will be assigned later as the object label distribution
                    else:
                        obj_prob = nn.functional.softmax(obj_output, dim=1).cpu().numpy()
                    hoi_probs = torch.sigmoid(hoi_output).cpu().numpy()
                    hoi_img_inds = hoi_infos[:, 0]
                    ho_pairs = hoi_infos[:, 1:]
                else:
                    hoi_probs = ho_pairs = hoi_img_inds = None
                    obj_prob = None

                if boxes_ext is not None:
                    im_scales = x.img_infos[:, 2].cpu().numpy()
                    boxes_ext = boxes_ext.cpu().numpy()
                    obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
                    obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
                    if obj_prob is None:
                        obj_prob = boxes_ext[:, 5:]  # this cannot be refined because of the lack of spatial relationships
                else:
                    obj_im_inds = obj_boxes = None
                return Prediction(obj_im_inds=obj_im_inds,
                                  obj_boxes=obj_boxes,
                                  obj_scores=obj_prob,
                                  hoi_img_inds=hoi_img_inds,
                                  ho_pairs=ho_pairs,
                                  hoi_scores=hoi_probs)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        raise NotImplementedError()

    def detection_and_gt_assignment(self, batch: Minibatch, predict):
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

        if not predict:
            boxes_ext_np, box_labels, box_feats, masks = self.box_gt_assignment(batch, boxes_ext_np, box_feats, masks, feat_map)
            hoi_im_ids, ho_pairs, hoi_labels = self.hoi_gt_assignments(batch, boxes_ext_np)  # FIXME magic constant
            assert hoi_im_ids.shape[0] == hoi_labels.shape[0] == ho_pairs.shape[0]
            assert box_labels.shape[0] == boxes_ext_np.shape[0] == box_feats.shape[0] == masks.shape[0]
            box_labels = torch.tensor(box_labels, device=masks.device)
            hoi_labels = torch.tensor(hoi_labels, device=masks.device)
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
                    return None, None, None, None, None, None, None

            # Sort by image
            inds = np.argsort(boxes_ext_np[:, 0]).astype(np.int64, copy=False)
            boxes_ext_np = boxes_ext_np[inds]
            box_feats = box_feats[inds]
            masks = masks[inds]

            hoi_im_ids, ho_pairs = self.get_all_pairs(boxes_ext_np)
            box_labels = hoi_labels = None
        boxes_ext = torch.tensor(boxes_ext_np, device=masks.device, dtype=torch.float32)

        if hoi_im_ids.size == 0:
            assert predict and not cfg.program.predcls
            return boxes_ext, box_feats, masks, None, None, None, None
        assert ho_pairs.shape[0] > 0

        # Note that box indices in `ho_pairs` are over all boxes, NOT relative to each specific image
        hoi_union_boxes = get_union_boxes(boxes_ext_np[:, 1:5], ho_pairs)
        union_boxes_feats = self.mask_rcnn.get_rois_feats(fmap=feat_map, rois=hoi_union_boxes)
        assert hoi_im_ids.shape[0] == union_boxes_feats.shape[0]
        hoi_infos = np.concatenate([hoi_im_ids[:, None], ho_pairs], axis=1).astype(np.int, copy=False)
        return boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels, hoi_labels

    def filter_and_map_to_hico(self, boxes_ext: np.ndarray, masks: torch.Tensor, box_feats: torch.Tensor):
        # Keep foreground object only
        class_scores = boxes_ext[:, 5:]
        classes = np.argmax(class_scores, axis=1)
        fg_inds = (classes > 0)
        assert fg_inds.shape[0] == boxes_ext.shape[0] == masks.shape[0]
        fg_inds = np.flatnonzero(fg_inds)  # this is needed for torch tensors
        boxes_ext_fg, masks, box_feats = boxes_ext[fg_inds, :], masks[fg_inds, :], box_feats[fg_inds, :]

        # Map from COCO classes to HICO ones
        coco_to_hico_mapping = np.array(self.dataset.coco_to_hico_mapping, dtype=np.int)
        boxes_ext_hico = boxes_ext_fg[:, np.concatenate([np.arange(5), 5 + coco_to_hico_mapping])]  # convert to Hico classes by swapping columns

        return boxes_ext_hico, masks, box_feats

    def get_all_pairs(self, boxes_ext, box_classes=None):
        # FIXME there is a significant overlap between this method and rel_gt_assignment. Merge.
        box_classes = box_classes or np.argmax(boxes_ext[:, 5:], axis=1)
        person_box_inds = (box_classes == self.dataset.person_class)

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
        if unmatched_gt_boxes_inds.size > 0:
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

        # Sort by image
        inds = np.argsort(boxes_ext[:, 0]).astype(np.int64, copy=False)
        boxes_ext = boxes_ext[inds]
        obj_labels = obj_labels[inds]
        box_feats = box_feats[inds]
        masks = masks[inds]

        return boxes_ext, obj_labels, box_feats, masks

    def hoi_gt_assignments(self, batch: Minibatch, boxes_ext_np, gt_match_iou_thr=0.5):
        gt_boxes, gt_box_im_ids, gt_obj_classes = batch.gt_boxes, batch.gt_box_im_ids, batch.gt_obj_classes
        gt_inters, gt_inters_im_ids = batch.gt_hois, batch.gt_hoi_im_ids
        predict_box_im_ids = boxes_ext_np[:, 0]
        predict_boxes = boxes_ext_np[:, 1:5]
        predict_box_classes = np.argmax(boxes_ext_np[:, 5:], axis=1)

        hois_ext = []
        num_box_seen = 0
        for im_id in np.unique(gt_box_im_ids):
            # Get image values
            predict_box_im_ids_i = (predict_box_im_ids == im_id)
            gt_box_im_ids_i = (gt_box_im_ids == im_id)
            assert np.any(predict_box_im_ids_i)

            gt_boxes_i = gt_boxes[gt_box_im_ids_i]
            gt_obj_classes_i = gt_obj_classes[gt_box_im_ids_i]
            gt_rels_i = gt_inters[gt_inters_im_ids == im_id]

            predict_boxes_i = predict_boxes[predict_box_im_ids_i]
            predict_box_labels_i = predict_box_classes[predict_box_im_ids_i]
            num_predict_boxes_i = predict_boxes_i.shape[0]

            # Find rel distribution
            iou_predict_to_gt_i = compute_ious(predict_boxes_i, gt_boxes_i)
            predict_gt_match_i = (predict_box_labels_i[:, None] == gt_obj_classes_i[None, :]) & (iou_predict_to_gt_i >= gt_match_iou_thr)

            hois_i = np.zeros((num_predict_boxes_i, num_predict_boxes_i, self.dataset.num_predicates))
            for from_gt_ind, rel_id, to_gt_ind in gt_rels_i:
                for from_predict_ind in np.flatnonzero(predict_gt_match_i[:, from_gt_ind]):
                    for to_predict_ind in np.flatnonzero(predict_gt_match_i[:, to_gt_ind]):
                        if from_predict_ind != to_predict_ind:
                            hois_i[from_predict_ind, to_predict_ind, rel_id] = 1.0

            ho_pairs_i = np.where(hois_i.any(axis=2))
            hois_i_ext = np.concatenate([np.full(ho_pairs_i.shape[0], fill_value=im_id),
                                         ho_pairs_i[0][:, None] + num_box_seen,
                                         ho_pairs_i[1][:, None] + num_box_seen,
                                         hois_i[ho_pairs_i]],
                                        axis=1)
            assert hois_i_ext.shape[0] > 0  # since GT boxes are added to predicted ones during training this cannot be empty

            hois_ext.append(hois_i_ext)
            num_box_seen += num_predict_boxes_i

        hois_ext = np.concatenate(hois_ext, axis=0)
        hoi_im_ids = hois_ext[:, 0]
        ho_pairs = hois_ext[:, [1, 2]]  # [sub_ind, obj_ind]
        hoi_labels = hois_ext[:, 3:]  # [pred]
        return hoi_im_ids, ho_pairs, hoi_labels
