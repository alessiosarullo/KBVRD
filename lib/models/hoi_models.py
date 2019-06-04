from lib.bbox_utils import compute_ious
from lib.dataset.utils import Splits
from lib.models.generic_model import GenericModel, Prediction
from lib.dataset.hicodet.pc_hicodet_split import PrecomputedMinibatch
from lib.models.containers import VisualOutput
from lib.models.hoi_branches import *
from lib.models.obj_branches import *


class OracleModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'oracle'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        raise NotImplementedError('This needs to be checked after refactors.')
        self.fake = torch.nn.Parameter(torch.from_numpy(np.array([1.])), requires_grad=True)
        self.iou_thresh = 0.5
        self.split = Splits.TEST
        self.perfect_detector = True

    def forward(self, x: PrecomputedMinibatch, inference=True, **kwargs):
        assert inference is True
        assert not self.training

        boxes_ext, box_feats, masks, union_boxes, union_boxes_feats, hoi_infos, box_labels, action_labels, hoi_labels = self.visual_module(x, True)
        im_scales = x.img_infos[:, 2].cpu().numpy()
        gt_entry = HicoDetSplitBuilder.get_split(HicoDetSplit, self.split).get_img_entry(x.other_ex_data[0]['index'], read_img=False)
        gt_boxes = gt_entry.gt_boxes * im_scales[0]
        gt_obj_classes = gt_entry.gt_obj_classes
        if self.perfect_detector:
            boxes_ext = torch.from_numpy(np.concatenate([np.zeros((gt_boxes.shape[0], 1)),
                                                         gt_boxes,
                                                         self.visual_module.one_hot_obj_labels(gt_obj_classes)
                                                         ], axis=1))
            hoi_infos = self.visual_module.get_all_pairs(boxes_ext[:, :5].detach().cpu().numpy(), gt_obj_classes)
            if hoi_infos.size == 0:
                hoi_infos = None

        if hoi_infos is not None:
            im_ids = np.unique(hoi_infos[:, 0])
            assert im_ids.size == 1 and im_ids == 0

            predict_boxes = boxes_ext[:, 1:5].detach().cpu().numpy()
            pred_gt_ious = compute_ious(predict_boxes, gt_boxes)

            pred_gt_best_match = np.argmax(pred_gt_ious, axis=1)  # type: np.ndarray
            box_labels = gt_obj_classes[pred_gt_best_match]  # assign the best match
            obj_output = torch.from_numpy(self.visual_module.one_hot_obj_labels(box_labels))

            pred_gt_ious_class_match = (box_labels[:, None] == gt_obj_classes[None, :])

            predict_ho_pairs = hoi_infos[:, 1:]
            gt_hois = gt_entry.gt_hois[:, [0, 2, 1]]

            action_labels = np.zeros((predict_ho_pairs.shape[0], self.dataset.num_predicates))
            for predict_idx, (ph, po) in enumerate(predict_ho_pairs):
                gt_pair_ious = np.zeros(gt_hois.shape[0])
                for gtidx, (gh, go, gi) in enumerate(gt_hois):
                    iou_h = pred_gt_ious[ph, gh]
                    iou_o = pred_gt_ious[po, go]
                    if pred_gt_ious_class_match[ph, gh] and pred_gt_ious_class_match[po, go]:
                        gt_pair_ious[gtidx] = min(iou_h, iou_o)
                if np.any(gt_pair_ious > self.iou_thresh):
                    gtidxs = (gt_pair_ious > self.iou_thresh)
                    action_labels[predict_idx, np.unique(gt_hois[gtidxs, 2])] = 1

            action_output = torch.from_numpy(action_labels)
        else:
            obj_output = action_output = boxes_ext = None

        if not inference:
            assert obj_output is not None and action_output is not None and box_labels is not None and action_labels is not None
            return obj_output, action_output, box_labels, action_labels
        else:
            return self._prepare_prediction(obj_output, action_output, hoi_infos, boxes_ext, im_scales=im_scales)

    def _forward(self):
        raise NotImplementedError()

    def _prepare_prediction(self, obj_output, action_output, hoi_output, hoi_infos, boxes_ext, im_scales):
        if hoi_infos is not None:
            assert obj_output is not None and action_output is not None and boxes_ext is not None
            obj_prob = obj_output.cpu().numpy()
            action_probs = action_output.cpu().numpy()
            ho_img_inds = hoi_infos[:, 0]
            ho_pairs = hoi_infos[:, 1:]
        else:
            action_probs = ho_pairs = ho_img_inds = None
            obj_prob = None

        if boxes_ext is not None:
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
                          ho_img_inds=ho_img_inds,
                          ho_pairs=ho_pairs,
                          action_scores=action_probs)


class ObjFGPredModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'objfgpred'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        self.fg_thr = 0.5
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim

        self.fg_obj_branch = SimpleObjBranch(input_dim=vis_feat_dim + self.dataset.num_object_classes)
        self.fg_obj_output_fc = nn.Linear(self.fg_obj_branch.output_dim, 1)

        self.obj_branch = SimpleObjBranch(input_dim=vis_feat_dim + self.dataset.num_object_classes)
        self.obj_output_fc = nn.Linear(self.obj_branch.output_dim, self.dataset.num_object_classes)

        self.act_branch = SimpleHoiBranch(self.visual_module.vis_feat_dim, self.obj_branch.output_dim, use_relu=cfg.model.relu)
        self.action_output_fc = nn.Linear(self.act_branch.output_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.action_output_fc.weight, gain=1.0)

    def forward(self, x: PrecomputedMinibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            vis_output = self.visual_module(x, inference)  # type: VisualOutput

            if vis_output.ho_infos is not None:
                obj_logits, action_logits, fg_obj_logits = self._forward(vis_output)
            else:
                obj_logits = action_logits = fg_obj_logits = None

            if not inference:
                box_labels = vis_output.box_labels  # type: torch.Tensor
                action_labels = vis_output.action_labels
                fg_box_labels = torch.cat([box_labels.new_ones((box_labels.shape[0], 1)),
                                           box_labels.new_zeros((fg_obj_logits.shape[0] - box_labels.shape[0], 1))], dim=0).float()

                losses = {'object_loss': nn.functional.cross_entropy(obj_logits, box_labels),
                          'fg_object_loss': nn.functional.binary_cross_entropy_with_logits(fg_obj_logits, fg_box_labels),
                          'action_loss': nn.functional.binary_cross_entropy_with_logits(action_logits, action_labels) * action_logits.shape[1]}
                return losses
            else:
                prediction = Prediction()

                if vis_output.ho_infos is not None:
                    assert action_logits is not None

                    prediction.ho_img_inds = vis_output.ho_infos[:, 0]
                    prediction.ho_pairs = vis_output.ho_infos[:, 1:]
                    prediction.obj_scores = nn.functional.softmax(obj_logits, dim=1).cpu().numpy()
                    prediction.action_scores = torch.sigmoid(action_logits).cpu().numpy()

                if vis_output.boxes_ext is not None:
                    boxes_ext = vis_output.boxes_ext.cpu().numpy()
                    im_scales = x.img_infos[:, 2].cpu().numpy()

                    obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
                    obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
                    prediction.obj_im_inds = obj_im_inds
                    prediction.obj_boxes = obj_boxes

                return prediction

    def _forward(self, vis_output: VisualOutput):
        if vis_output.box_labels is not None:
            bg_boxes_ext, bg_box_feats, bg_masks = vis_output.filter_boxes()
            boxes_ext = vis_output.boxes_ext
            box_feats = vis_output.box_feats
            masks = vis_output.masks

            fg_and_bg_boxes_ext = torch.cat((boxes_ext, bg_boxes_ext), dim=0)
            fg_and_bg_box_feats = torch.cat((box_feats, bg_box_feats), dim=0)

            fg_obj_repr = self.fg_obj_branch(fg_and_bg_boxes_ext, fg_and_bg_box_feats)
            fg_obj_logits = self.fg_obj_output_fc(fg_obj_repr)
        else:
            fg_and_bg_boxes_ext = vis_output.boxes_ext
            fg_and_bg_box_feats = vis_output.box_feats

            fg_obj_repr = self.fg_obj_branch(fg_and_bg_boxes_ext, fg_and_bg_box_feats)
            fg_obj_logits = self.fg_obj_output_fc(fg_obj_repr)
            fg_score = torch.sigmoid(fg_obj_logits.squeeze(dim=1))
            vis_output.filter_boxes(valid_box_mask=(fg_score >= self.fg_thr))

            if vis_output.boxes_ext is None:
                return None, None, None

            boxes_ext = vis_output.boxes_ext
            box_feats = vis_output.box_feats
            masks = vis_output.masks

        obj_repr = self.obj_branch(boxes_ext, box_feats)
        obj_logits = self.obj_output_fc(obj_repr)

        if vis_output.ho_infos is None:
            assert vis_output.box_labels is None
            return obj_logits, None, None

        union_boxes_feats = vis_output.hoi_union_boxes_feats
        hoi_infos = torch.tensor(vis_output.ho_infos, device=masks.device)

        act_repr = self.act_branch(obj_repr, union_boxes_feats, hoi_infos)
        action_logits = self.action_output_fc(act_repr)

        return obj_logits, action_logits, fg_obj_logits


class FuncGenModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'fgen'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        self.fc1_dim = 1024
        self.fc2_dim = 512
        self.word_emb_dim = 300
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim

        word_embs = WordEmbeddings(source='glove', dim=self.word_emb_dim)
        obj_word_embs = word_embs.get_embeddings(dataset.objects)
        self.obj_word_embs = nn.Embedding.from_pretrained(torch.from_numpy(obj_word_embs), freeze=True)

        self.act_branch = nn.Sequential(*[nn.Linear(vis_feat_dim + self.word_emb_dim + 14, self.fc1_dim),  # 14 = # geometric features
                                          nn.ReLU(inplace=True),
                                          nn.Linear(self.fc1_dim, self.fc2_dim),
                                          nn.ReLU(inplace=True),
                                          ])
        self.act_output_fc = nn.Linear(self.fc2_dim, dataset.num_predicates, bias=True)

    def forward(self, x: PrecomputedMinibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            vis_output = self.visual_module(x, inference)  # type: VisualOutput

            if vis_output.ho_infos is not None:
                vis_output.minibatch = x
                action_output = self._forward(vis_output, x)
            else:
                assert inference
                action_output = None

            if not inference:
                action_labels = vis_output.action_labels
                losses = {'action_loss': nn.functional.binary_cross_entropy_with_logits(action_output, action_labels) * action_output.shape[1]}
                return losses
            else:
                prediction = Prediction()

                if vis_output.boxes_ext is not None:
                    boxes_ext = vis_output.boxes_ext.cpu().numpy()
                    im_scales = x.img_infos[:, 2].cpu().numpy()

                    obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
                    obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
                    prediction.obj_im_inds = obj_im_inds
                    prediction.obj_boxes = obj_boxes
                    prediction.obj_scores = boxes_ext[:, 5:]

                    if vis_output.ho_infos is not None:
                        assert action_output is not None

                        prediction.ho_img_inds = vis_output.ho_infos[:, 0]
                        prediction.ho_pairs = vis_output.ho_infos[:, 1:]
                        prediction.action_scores = torch.sigmoid(action_output).cpu().numpy()

                return prediction

    # noinspection PyMethodOverriding
    def _forward(self, vis_output: VisualOutput, batch: PrecomputedMinibatch):
        if vis_output.box_labels is not None:
            vis_output.filter_boxes()
        boxes_ext = vis_output.boxes_ext
        box_feats = vis_output.box_feats
        masks = vis_output.masks
        hoi_infos = torch.tensor(vis_output.ho_infos, device=masks.device)

        im_sizes = torch.tensor(np.array([d['im_size'][::-1] * d['im_scale'] for d in batch.other_ex_data]).astype(np.float32), device=masks.device)
        im_areas = im_sizes.prod(dim=1)

        # Needed for numerical errors. Also when assigning GT to detections this is not guaranteed to be true.
        # FIXME this should be needed, remove
        box_im_inds = boxes_ext[:, 0].long()
        box_im_sizes = im_sizes[box_im_inds, :]
        boxes_ext[:, 3] = torch.min(boxes_ext[:, 3], box_im_sizes[:, 0])
        boxes_ext[:, 4] = torch.min(boxes_ext[:, 4], box_im_sizes[:, 1])

        norm_boxes = boxes_ext[:, 1:5] / box_im_sizes.repeat(1, 2)
        assert (0 <= norm_boxes).all(), \
            (box_im_inds.detach().cpu().numpy(), boxes_ext[:, 1:5].detach().cpu().numpy(),
             im_sizes.detach().cpu().numpy(), norm_boxes.detach().cpu().numpy())
        # norm_boxes.clamp_(max=1)  # Needed for numerical errors
        assert (norm_boxes <= 1).all(), \
            (box_im_inds.detach().cpu().numpy(), boxes_ext[:, 1:5].detach().cpu().numpy(),
             im_sizes.detach().cpu().numpy(), norm_boxes.detach().cpu().numpy())

        box_widths = boxes_ext[:, 3] - boxes_ext[:, 1]
        box_heights = boxes_ext[:, 4] - boxes_ext[:, 2]
        norm_box_areas = box_widths * box_heights / im_areas[box_im_inds]
        assert (0 < norm_box_areas).all(), \
            (box_im_inds.detach().cpu().numpy(), boxes_ext[:, 1:5].detach().cpu().numpy(), norm_box_areas.detach().cpu().numpy())
        norm_box_areas.clamp_(max=1)  # Same as above
        assert (norm_box_areas <= 1).all(), \
            (box_im_inds.detach().cpu().numpy(), boxes_ext[:, 1:5].detach().cpu().numpy(), norm_box_areas.detach().cpu().numpy())

        hum_inds = hoi_infos[:, 1]
        obj_inds = hoi_infos[:, 2]
        obj_widths = box_widths[obj_inds]
        obj_heights = box_widths[obj_inds]

        h_dist = (boxes_ext[hum_inds, 1] - boxes_ext[obj_inds, 1]) / obj_widths
        v_dist = (boxes_ext[hum_inds, 2] - boxes_ext[obj_inds, 2]) / obj_heights

        h_ratio = (box_widths[hum_inds] / obj_widths).log()
        v_ratio = (box_widths[hum_inds] / obj_heights).log()

        geo_feats = torch.cat([norm_boxes[hum_inds, :],
                               norm_box_areas[hum_inds, None],
                               norm_boxes[obj_inds, :],
                               norm_box_areas[obj_inds, None],
                               h_dist[:, None], v_dist[:, None], h_ratio[:, None], v_ratio[:, None]
                               ], dim=1)

        obj_word_embs = self.obj_word_embs(boxes_ext[obj_inds, 5:].argmax(dim=1))

        hum_repr = box_feats[hum_inds, :]

        act_repr = self.act_branch(torch.cat([geo_feats, obj_word_embs, hum_repr], dim=1))
        action_logits = self.act_output_fc(act_repr)

        return action_logits


class BaseModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'base'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        self.act_repr_dim = 600
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim

        self.ho_subj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, self.act_repr_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5),
                                                nn.Linear(self.act_repr_dim, self.act_repr_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5),
                                                ])
        nn.init.xavier_normal_(self.ho_subj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.ho_subj_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('relu'))

        self.ho_obj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, self.act_repr_dim),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(0.5),
                                               nn.Linear(self.act_repr_dim, self.act_repr_dim),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(0.5),
                                               ])
        nn.init.xavier_normal_(self.ho_obj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.ho_obj_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('relu'))

        self.union_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim, self.act_repr_dim),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(0.5),
                                              nn.Linear(self.act_repr_dim, self.act_repr_dim),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(0.5),
                                              ])
        nn.init.xavier_normal_(self.union_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.union_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('relu'))

        self.act_output_fc = nn.Linear(self.act_repr_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.act_output_fc.weight, gain=1.0)

    def _forward(self, vis_output: VisualOutput):
        if vis_output.box_labels is not None:
            vis_output.filter_boxes()
        boxes_ext = vis_output.boxes_ext
        box_feats = vis_output.box_feats
        masks = vis_output.masks
        union_boxes_feats = vis_output.hoi_union_boxes_feats
        hoi_infos = torch.tensor(vis_output.ho_infos, device=masks.device)

        box_info = torch.cat([box_feats, boxes_ext[:, 5:]], dim=1)

        ho_subj_repr = self.ho_subj_repr_mlp(box_info[hoi_infos[:, 1], :])
        ho_obj_repr = self.ho_obj_repr_mlp(box_info[hoi_infos[:, 2], :])
        union_repr = self.union_repr_mlp(union_boxes_feats)
        act_repr = union_repr + ho_subj_repr + ho_obj_repr

        action_logits = self.act_output_fc(act_repr)

        return action_logits


class HoiBaseModel(BaseModel):
    @classmethod
    def get_cline_name(cls):
        return 'hoibase'

    def forward(self, x: PrecomputedMinibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            vis_output = self.visual_module(x, inference)  # type: VisualOutput

            if vis_output.ho_infos is not None:
                action_output = self._forward(vis_output)
            else:
                assert inference
                action_output = None

            if not inference:
                action_labels = vis_output.action_labels
                losses = {'action_loss': nn.functional.binary_cross_entropy_with_logits(action_output, action_labels) * action_output.shape[1]}
                return losses
            else:
                prediction = Prediction()

                if vis_output.boxes_ext is not None:
                    boxes_ext = vis_output.boxes_ext.cpu().numpy()
                    im_scales = x.img_infos[:, 2].cpu().numpy()

                    obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
                    obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
                    prediction.obj_im_inds = obj_im_inds
                    prediction.obj_boxes = obj_boxes
                    prediction.obj_scores = boxes_ext[:, 5:]

                    if vis_output.ho_infos is not None:
                        assert action_output is not None

                        prediction.ho_img_inds = vis_output.ho_infos[:, 0]
                        prediction.ho_pairs = vis_output.ho_infos[:, 1:]
                        prediction.action_scores = torch.sigmoid(action_output).cpu().numpy()

                        ho_obj_probs = prediction.obj_scores[vis_output.ho_infos[:, 2], :]
                        hoi_probs = np.zeros((action_output.shape[0], self.dataset.num_interactions))
                        for iid, (pid, oid) in enumerate(self.dataset.interactions):
                            hoi_probs[:, iid] = ho_obj_probs[:, oid] * prediction.action_scores[:, pid]
                        prediction.hoi_scores = hoi_probs

                return prediction


class KatoModel(GenericModel):
    # FIXME?
    @classmethod
    def get_cline_name(cls):
        return 'kato'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        raise NotImplementedError('This needs to be checked after refactors.')
        self.obj_branch = SimpleObjBranch(input_dim=self.visual_module.vis_feat_dim + self.dataset.num_object_classes)
        self.obj_output_fc = nn.Linear(self.obj_branch.output_dim, self.dataset.num_object_classes)

        self.hoi_branch = KatoGCNBranch(self.visual_module.vis_feat_dim, self.obj_branch.output_dim, dataset)

    def _forward(self):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        obj_repr = self.obj_branch(boxes_ext, box_feats)
        obj_logits = self.obj_output_fc(obj_repr)

        hoi_obj_logits, action_logits = self.hoi_branch(obj_repr, union_boxes_feats, hoi_infos)

        return obj_logits, action_logits


class PeyreModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'peyre'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        raise NotImplementedError('This needs to be checked after refactors.')
        self.hoi_branch = PeyreEmbsimBranch(self.visual_module.vis_feat_dim, dataset)

    def get_losses(self, x, **kwargs):
        y = self(x, inference=False, **kwargs)

        hoi_subj_logits, hoi_subj_labels = y[0]
        subj_labels_1hot = hoi_subj_labels.new_zeros((hoi_subj_labels.shape[0], self.dataset.num_object_classes)).float()
        subj_labels_1hot[torch.arange(subj_labels_1hot.shape[0]), hoi_subj_labels] = 1
        hoi_subj_loss = nn.functional.binary_cross_entropy_with_logits(hoi_subj_logits, subj_labels_1hot) * self.dataset.num_object_classes

        hoi_obj_logits, hoi_obj_labels = y[1]
        obj_labels_1hot = hoi_obj_labels.new_zeros((hoi_obj_labels.shape[0], self.dataset.num_object_classes)).float()
        obj_labels_1hot[torch.arange(obj_labels_1hot.shape[0]), hoi_obj_labels] = 1
        hoi_obj_loss = nn.functional.binary_cross_entropy_with_logits(hoi_obj_logits, obj_labels_1hot) * self.dataset.num_object_classes

        hoi_act_logits, action_labels = y[2]
        act_loss = nn.functional.binary_cross_entropy_with_logits(hoi_act_logits, action_labels) * self.dataset.num_predicates

        hoi_logits, hoi_labels = y[3]
        hoi_loss = nn.functional.binary_cross_entropy_with_logits(hoi_logits, hoi_labels) * self.dataset.num_interactions

        return {'hoi_subj_loss': hoi_subj_loss, 'hoi_obj_loss': hoi_obj_loss, 'action_loss': act_loss, 'hoi_loss': hoi_loss}

    def forward(self, x, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            boxes_ext, box_feats, masks, union_boxes, union_boxes_feats, hoi_infos, box_labels, action_labels, hoi_labels = \
                self.visual_module(x, inference)
            # `hoi_infos` is an R x 3 NumPy array where each column is [image ID, subject index, object index].
            # Masks are floats at this point.

            if hoi_infos is not None:
                logits = self._forward()
                hoi_subj_logits, hoi_obj_logits, hoi_act_logits, hoi_logits = logits
            else:
                hoi_subj_logits = hoi_obj_logits = hoi_act_logits = hoi_logits = None

            if not inference:
                assert all([x is not None for x in (box_labels, action_labels, hoi_labels)])

                hoi_subj_labels = box_labels[hoi_infos[:, 1]]
                # assert torch.unique(hoi_subj_labels).shape[0] == 1, hoi_subj_labels  # Doesn't work because of negative sampling. FIXME?

                hoi_obj_labels = box_labels[hoi_infos[:, 2]]
                return (hoi_subj_logits, hoi_subj_labels), (hoi_obj_logits, hoi_obj_labels), (hoi_act_logits, action_labels), (hoi_logits, hoi_labels)
            else:
                im_scales = x.img_infos[:, 2].cpu().numpy()
                if hoi_logits is None:
                    hoi_output = None
                else:
                    hoi_output = np.empty([hoi_logits.shape[0], self.dataset.num_interactions])
                    for iid, (pid, oid) in enumerate(self.dataset.interactions):
                        hoi_subj_prob = torch.sigmoid(hoi_subj_logits[:, self.dataset.human_class])
                        hoi_obj_prob = torch.sigmoid(hoi_obj_logits[:, oid])
                        hoi_act_prob = torch.sigmoid(hoi_act_logits[:, pid])
                        hoi_prob = torch.sigmoid(hoi_logits[:, iid])
                        hoi_output[:, iid] = hoi_subj_prob * hoi_obj_prob * hoi_act_prob * hoi_prob
                return self._prepare_prediction(None, None, hoi_output, hoi_infos, boxes_ext, im_scales)

    def _forward(self):
        hoi_subj_logits, hoi_obj_logits, hoi_act_logits, hoi_logits = self.hoi_branch(boxes_ext, box_feats, hoi_infos)
        return hoi_subj_logits, hoi_obj_logits, hoi_act_logits, hoi_logits

    def _prepare_prediction(self, obj_output, action_output, hoi_probs, hoi_infos, boxes_ext, im_scales):
        assert obj_output is None and action_output is None

        ho_pairs = ho_img_inds = None
        if hoi_infos is not None:
            assert boxes_ext is not None
            assert hoi_probs is not None
            ho_img_inds = hoi_infos[:, 0]
            ho_pairs = hoi_infos[:, 1:]

        if boxes_ext is not None:
            boxes_ext = boxes_ext.cpu().numpy()
            obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
            obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
        else:
            obj_im_inds = obj_boxes = None
        return Prediction(obj_im_inds=obj_im_inds,
                          obj_boxes=obj_boxes,
                          obj_scores=None,
                          ho_img_inds=ho_img_inds,
                          ho_pairs=ho_pairs,
                          action_scores=None,
                          hoi_scores=hoi_probs)
