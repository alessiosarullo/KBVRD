import math

import numpy as np
import torch
from torch import nn as nn
from torch.nn.utils.rnn import PackedSequence

from config import cfg
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.dataset.utils import get_counts
from lib.dataset.word_embeddings import WordEmbeddings
from lib.knowledge_extractors.conceptnet_knowledge_extractor import ConceptnetKnowledgeExtractor
from lib.knowledge_extractors.imsitu_knowledge_extractor import ImSituKnowledgeExtractor
from lib.models.abstract_model import AbstractHOIBranch


class NMotifsHOIBranch(AbstractHOIBranch):
    def __init__(self, visual_feats_dim, obj_feat_dim, dataset, **kwargs):
        # FIXME ugliness
        from lib.models.highway_lstm_cuda.alternating_highway_lstm import AlternatingHighwayLSTM
        from lib.models.nmotifs.lincontext import sort_rois
        self.sort_rois = sort_rois

        self.word_emb_dim = 200
        self.edge_ctx_num_layers = 4
        self.rnn_hidden_dim = 256
        self.order = 'leftright'
        self.dropout_rate = 0.1
        super().__init__(**kwargs)
        self.hoi_repr_dim = visual_feats_dim

        self.word_embs = WordEmbeddings(source='glove', dim=self.word_emb_dim)
        self.obj_word_embs = torch.nn.Embedding.from_pretrained(torch.from_numpy(self.word_embs.get_embeddings(dataset.objects)), freeze=True)

        hoi_input_dim = obj_feat_dim
        self.hoi_obj_birnn = AlternatingHighwayLSTM(input_size=self.word_emb_dim + hoi_input_dim,
                                                    hidden_size=self.rnn_hidden_dim,
                                                    num_layers=self.edge_ctx_num_layers,
                                                    recurrent_dropout_probability=self.dropout_rate)
        self.post_lstm = nn.Linear(self.rnn_hidden_dim, self.hoi_repr_dim * 2)
        torch.nn.init.normal_(self.post_lstm.weight, mean=0, std=10 * math.sqrt(1.0 / self.rnn_hidden_dim))
        torch.nn.init.zeros_(self.post_lstm.bias)

    @property
    def output_dim(self):
        return self.hoi_repr_dim

    def _forward(self, boxes_ext, box_repr, union_boxes_feats, hoi_infos, box_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        subj_inds = hoi_infos[:, 1]
        dobj_inds = hoi_infos[:, 2]

        # FIXME this doesn't use spatial context
        obj_classes = box_labels if box_labels is not None else torch.argmax(boxes_ext[:, 5:], dim=1)
        hoi_input_obj_repr = torch.cat((self.obj_word_embs(obj_classes), box_repr), dim=1)
        perm, inv_perm, ls_transposed = self.sort_rois(self.order, box_im_ids, box_priors=boxes_ext[:, 1:5])
        hoi_output_obj_repr = self.hoi_obj_birnn(PackedSequence(hoi_input_obj_repr[perm], ls_transposed))[0]
        hoi_output_obj_repr = hoi_output_obj_repr[inv_perm]

        hoi_ho_repr = self.post_lstm(hoi_output_obj_repr)
        hoi_ho_repr = hoi_ho_repr.view(hoi_ho_repr.shape[0], 2, -1)  # Split into subject and object representations
        subj_repr = hoi_ho_repr[:, 0][subj_inds]
        dobj_repr = hoi_ho_repr[:, 1][dobj_inds]
        hoi_repr = subj_repr * dobj_repr * union_boxes_feats

        return hoi_repr


class SimpleHoiBranch(AbstractHOIBranch):
    def __init__(self, visual_feats_dim, obj_repr_dim, **kwargs):
        # TODO docs and FIXME comments
        self.hoi_repr_dim = 600
        super().__init__(**kwargs)

        self.union_repr_fc = nn.Linear(visual_feats_dim, self.hoi_repr_dim)
        nn.init.xavier_normal_(self.union_repr_fc.weight, gain=1.0)

        self.hoi_obj_repr_fc = nn.Linear(obj_repr_dim, self.hoi_repr_dim)
        nn.init.xavier_normal_(self.hoi_obj_repr_fc.weight, gain=1.0)

    @property
    def output_dim(self):
        return self.hoi_repr_dim

    def _forward(self, boxes_ext, obj_repr, union_boxes_feats, hoi_infos, obj_logits, box_labels=None):
        hoi_obj_repr = self.hoi_obj_repr_fc(obj_repr[hoi_infos[:, 2], :])
        # union_repr = union_boxes_feats
        union_repr = self.union_repr_fc(union_boxes_feats)
        hoi_repr = union_repr + hoi_obj_repr
        return hoi_repr


class HoiPriorBranch(AbstractHOIBranch):
    def __init__(self, dataset: HicoDetInstanceSplit, hoi_repr_dim, **kwargs):
        super().__init__(**kwargs)

        # Freq bias
        freqs = []
        if cfg.model.freq_bias:
            int_counts = get_counts(dataset=dataset)
            freqs.append(int_counts)
        # Possibly add here other priors

        if freqs:
            self.bias_priors = nn.ModuleList()
            for fmat in freqs:
                priors = fmat / np.maximum(1, np.sum(fmat, axis=1, keepdims=True))
                self.bias_priors.append(torch.nn.Embedding.from_pretrained(torch.from_numpy(priors).float(), freeze=not cfg.model.train_prior))

            if cfg.model.prior_att:
                self.prior_source_attention = nn.Sequential(nn.Linear(hoi_repr_dim, len(self.bias_priors)),
                                                            nn.Sigmoid())
            else:
                self.prior_source_attention = None
        else:  # no actual refinement
            self.bias_priors = None

    def _forward(self, hoi_logits, hoi_repr, boxes_ext, hoi_infos, box_labels=None):
        if self.bias_priors:
            obj_classes = box_labels if box_labels is not None else torch.argmax(boxes_ext[:, 5:], dim=1)
            hoi_obj_classes = obj_classes[hoi_infos[:, 2]].detach()

            priors = torch.stack([prior(hoi_obj_classes) for prior in self.bias_priors], dim=0).clamp(min=1e-3)  # FIXME magic constant

            if self.prior_source_attention is not None:
                src_att = self.prior_source_attention(hoi_repr)
                prior_contribution = (src_att.t().unsqueeze(dim=2) * priors).sum(dim=0)
                self.values_to_monitor['hoi_attention'] = src_att.detach().cpu().numpy()
            else:
                prior_contribution = priors.sum(dim=0)
            hoi_logits += prior_contribution.log()
        return hoi_logits


class HoiEmbsimBranch(AbstractHOIBranch):
    def __init__(self, visual_feats_dim, dataset: HicoDetInstanceSplit, **kwargs):
        # TODO docs and FIXME comments
        self.word_emb_dim = 300
        super().__init__(**kwargs)
        self.num_objects = dataset.num_object_classes
        self.num_predicates = dataset.num_predicates

        self.word_embs = WordEmbeddings(source='glove', dim=self.word_emb_dim)
        obj_word_embs = self.word_embs.get_embeddings(dataset.objects)
        pred_word_embs = self.word_embs.get_embeddings(dataset.predicates)

        interactions = dataset.hicodet.interactions  # each is [p, o]
        interaction_embs = np.concatenate([pred_word_embs[interactions[:, 0]],
                                           obj_word_embs[interactions[:, 1]]], axis=1)
        num_interactions = interactions.shape[0]
        assert num_interactions == 600
        interactions_to_obj = np.zeros((num_interactions, dataset.num_object_classes))
        interactions_to_obj[np.arange(num_interactions), interactions[:, 1]] = 1
        interactions_to_obj /= np.maximum(1, interactions_to_obj.sum(axis=0, keepdims=True))
        interactions_to_preds = np.zeros((num_interactions, dataset.num_predicates))
        interactions_to_preds[np.arange(num_interactions), interactions[:, 0]] = 1
        interactions_to_preds /= np.maximum(1, interactions_to_preds.sum(axis=0, keepdims=True))

        self.interaction_embs = nn.Parameter(torch.from_numpy(interaction_embs.T), requires_grad=False)
        self.interactions_to_obj = nn.Parameter(torch.from_numpy(interactions_to_obj).float(), requires_grad=False)
        self.interactions_to_preds = nn.Parameter(torch.from_numpy(interactions_to_preds).float(), requires_grad=False)
        self.op_cossim = torch.nn.CosineSimilarity(dim=1)

        self.obj_vis_to_emb_fc = nn.Sequential(nn.Linear(visual_feats_dim, 2 * self.word_emb_dim),
                                               nn.ReLU(),
                                               nn.Linear(2 * self.word_emb_dim, self.word_emb_dim))
        nn.init.xavier_normal_(self.obj_vis_to_emb_fc[0].weight, gain=1.0)
        nn.init.xavier_normal_(self.obj_vis_to_emb_fc[2].weight, gain=1.0)
        self.pred_vis_to_emb_fc = nn.Sequential(nn.Linear(visual_feats_dim, 2 * self.word_emb_dim),
                                                nn.ReLU(),
                                                nn.Linear(2 * self.word_emb_dim, self.word_emb_dim))
        nn.init.xavier_normal_(self.pred_vis_to_emb_fc[0].weight, gain=1.0)
        nn.init.xavier_normal_(self.pred_vis_to_emb_fc[2].weight, gain=1.0)

        # self.obj_output_fc = nn.Linear(self.obj_branch.repr_dim, self.dataset.num_object_classes)
        # torch.nn.init.xavier_normal_(self.obj_output_fc.weight, gain=1.0)
        # self.hoi_output_fc = nn.Linear(self.hoi_branch.output_dim, dataset.num_predicates, bias=True)
        # torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

    def _forward(self, union_box_feats, box_feats, hoi_infos):
        obj_embs = self.obj_vis_to_emb_fc(box_feats)
        pred_embs = self.pred_vis_to_emb_fc(union_box_feats)

        op_embs = torch.cat([pred_embs, obj_embs[hoi_infos[:, 2], :]], dim=1)
        op_sims = self.op_cossim(op_embs.unsqueeze(dim=2), self.interaction_embs.unsqueeze(dim=0))

        hoi_obj_logits = op_sims @ self.interactions_to_obj
        hoi_logits = op_sims @ self.interactions_to_preds

        return hoi_logits, hoi_obj_logits


class PeyreEmbsimBranch(AbstractHOIBranch):
    def __init__(self, visual_feats_dim, dataset: HicoDetInstanceSplit, **kwargs):
        self.word_emb_dim = 300
        super().__init__(**kwargs)

        self.word_embs = WordEmbeddings(source='glove', dim=self.word_emb_dim)
        obj_word_embs = self.word_embs.get_embeddings(dataset.objects)
        pred_word_embs = self.word_embs.get_embeddings(dataset.predicates)

        self.obj_word_embs = nn.Parameter(torch.from_numpy(obj_word_embs), requires_grad=False)
        self.pred_word_embs = nn.Parameter(torch.from_numpy(pred_word_embs), requires_grad=False)

        output_dim = 1024
        appearance_dim = 300
        spatial_dim = 400
        self.vis_to_app_mlps = nn.ModuleDict({k: nn.Linear(visual_feats_dim, appearance_dim) for k in ['sub', 'obj']})
        self.spatial_mlp = nn.Sequential(nn.Linear(8, spatial_dim),
                                         nn.Linear(spatial_dim, spatial_dim))
        self.app_to_repr_mlps = nn.ModuleDict({k: nn.Sequential(nn.Linear(appearance_dim, output_dim),
                                                                nn.ReLU(),
                                                                nn.Dropout(p=0.5),
                                                                nn.Linear(output_dim, output_dim)) for k in ['sub', 'obj']})
        self.app_to_repr_mlps['pred'] = nn.Sequential(nn.Linear(appearance_dim * 2 + spatial_dim, output_dim),
                                                      nn.ReLU(),
                                                      nn.Dropout(p=0.5),
                                                      nn.Linear(output_dim, output_dim))
        self.wemb_to_repr_mlps = nn.ModuleDict({k: nn.Sequential(nn.Linear(self.word_emb_dim, output_dim),
                                                                 nn.ReLU(),
                                                                 nn.Linear(output_dim, output_dim)) for k in ['sub', 'pred', 'obj']})

    def _forward(self, boxes_ext, box_feats, hoi_infos):
        boxes = boxes_ext[:, 1:5]
        hoi_hum_inds = hoi_infos[:, 1]
        hoi_obj_inds = hoi_infos[:, 2]
        union_boxes = torch.cat([
            torch.min(boxes[:, :2][hoi_hum_inds], boxes[:, :2][hoi_obj_inds]),
            torch.max(boxes[:, 2:][hoi_hum_inds], boxes[:, 2:][hoi_obj_inds]),
        ], dim=1)

        union_areas = (union_boxes[:, 2:] - union_boxes[:, :2]).prod(dim=1, keepdim=True)
        union_origin = union_boxes[:, :2].repeat(1, 2)
        hoi_hum_spatial_info = (boxes[hoi_hum_inds, :] - union_origin) / union_areas
        hoi_obj_spatial_info = (boxes[hoi_obj_inds, :] - union_origin) / union_areas
        spatial_info = self.spatial_mlp(torch.cat([hoi_hum_spatial_info, hoi_obj_spatial_info], dim=1))

        hoi_hum_appearance = self.vis_to_app_mlps['sub'](box_feats)[hoi_hum_inds, :]
        obj_appearance = self.vis_to_app_mlps['obj'](box_feats)
        hoi_obj_appearance = self.vis_to_app_mlps['obj'](box_feats)[hoi_obj_inds, :]

        obj_repr = self.app_to_repr_mlps['obj'](obj_appearance)
        pred_repr = self.app_to_repr_mlps['pred'](torch.cat([hoi_hum_appearance, hoi_obj_appearance, spatial_info], dim=1))

        obj_emb = self.wemb_to_repr_mlps['obj'](self.obj_word_embs)
        pred_emb = self.wemb_to_repr_mlps['pred'](self.pred_word_embs)

        obj_logits = obj_repr @ obj_emb.t()
        hoi_logits = pred_repr @ pred_emb.t()

        return obj_logits, hoi_logits


class HoiMemGCBranch(AbstractHOIBranch):
    def __init__(self, visual_feats_dim, dataset: HicoDetInstanceSplit, **kwargs):
        self.word_emb_dim = 300
        super().__init__(**kwargs)

        self.word_embs = WordEmbeddings(source='glove', dim=self.word_emb_dim)
        obj_word_embs = self.word_embs.get_embeddings(dataset.objects)
        pred_word_embs = self.word_embs.get_embeddings(dataset.predicates)

        op_adj_mats = []
        if cfg.model.use_ds:
            ds_counts = get_counts(dataset=dataset)
            ds_counts[:, 0] = 0  # exclude null interaction
            op_adj_mats.append(np.minimum(1, ds_counts))  # only check if the pair exists (>=1 occurrence) or not (0 occurrences)
        if cfg.model.use_imsitu:
            imsitu_counts = ImSituKnowledgeExtractor().extract_freq_matrix(dataset)
            imsitu_counts[:, 0] = 0  # exclude null interaction
            op_adj_mats.append(np.minimum(1, imsitu_counts))  # only check if the pair exists (>=1 occurrence) or not (0 occurrences)
        if cfg.model.use_cnet:
            cnet_counts = ConceptnetKnowledgeExtractor().extract_freq_matrix(dataset=dataset.hicodet)
            cnet_counts[:, 0] = 0  # exclude null interaction
            op_adj_mats.append(np.minimum(1, cnet_counts))  # only check if the pair exists (>=1 occurrence) or not (0 occurrences)

        assert op_adj_mats
        op_adj_mats = np.stack(op_adj_mats, axis=2)  # O x P x S
        op_adj_mat = np.sum(op_adj_mats, axis=2)  # O x P
        self.op_adj_mat = torch.nn.Parameter(torch.from_numpy(op_adj_mat).float().clamp(min=1e-2), requires_grad=True)

        # num_interactions = (op_adj_mat > 0).sum()
        # interactions_to_obj = np.zeros((num_interactions, dataset.num_object_classes))
        # interactions_to_obj[np.arange(num_interactions), np.where(op_adj_mat)[0]] = 1
        # # interactions_to_obj /= np.maximum(1, interactions_to_obj.sum(axis=0, keepdims=True))
        # interactions_to_preds = np.zeros((num_interactions, dataset.num_predicates))
        # interactions_to_preds[np.arange(num_interactions), np.where(op_adj_mat)[1]] = 1
        # # interactions_to_preds /= np.maximum(1, interactions_to_preds.sum(axis=0, keepdims=True))
        # self.interactions_to_obj = nn.Parameter(torch.from_numpy(interactions_to_obj).float(), requires_grad=False)
        # self.interactions_to_preds = nn.Parameter(torch.from_numpy(interactions_to_preds).float(), requires_grad=False)

        self.op_embs = torch.nn.Parameter(torch.from_numpy(obj_word_embs[:, None, :] + pred_word_embs[None, :, :]).float(), requires_grad=False)

        # self.obj_prob_fc = nn.Sequential(nn.Linear(dataset.num_object_classes, dataset.num_object_classes),
        #                                  torch.Sigmoid())
        # nn.init.xavier_normal_(self.obj_prob_fc[0].weight, gain=1.0)
        # self.hoi_prob_fc = nn.Sequential(nn.Linear(dataset.num_predicates, dataset.num_predicates),
        #                                  torch.Sigmoid())
        # nn.init.xavier_normal_(self.hoi_prob_fc[0].weight, gain=1.0)

        self.emb_readout_mlp = nn.Sequential(nn.Linear(self.op_embs.shape[2], self.word_emb_dim),
                                             nn.ReLU(),
                                             nn.Linear(self.word_emb_dim, self.word_emb_dim))
        nn.init.xavier_normal_(self.emb_readout_mlp[0].weight, gain=1.0)
        nn.init.xavier_normal_(self.emb_readout_mlp[2].weight, gain=1.0)

        self.pred_output_mlp = nn.Sequential(nn.Linear(self.word_emb_dim, self.word_emb_dim),
                                             nn.ReLU(),
                                             nn.Linear(self.word_emb_dim, 1))
        nn.init.xavier_normal_(self.pred_output_mlp[0].weight, gain=1.0)
        nn.init.xavier_normal_(self.pred_output_mlp[2].weight, gain=1.0)

        self.obj_output_mlp = nn.Sequential(nn.Linear(self.word_emb_dim, self.word_emb_dim),
                                            nn.ReLU(),
                                            nn.Linear(self.word_emb_dim, 1))
        nn.init.xavier_normal_(self.obj_output_mlp[0].weight, gain=1.0)
        nn.init.xavier_normal_(self.obj_output_mlp[2].weight, gain=1.0)

    def _forward(self, hoi_logits, obj_logits, union_box_feats, hoi_infos):
        obj_prediction = torch.sigmoid(obj_logits)
        hoi_prediction = torch.sigmoid(hoi_logits)

        obj_att = obj_prediction[hoi_infos[:, 2], :]  # N x O
        pred_att = hoi_prediction  # N x P
        joint_att = obj_att.unsqueeze(dim=2) * pred_att.unsqueeze(dim=1)  # N x O x P
        adj_joint_att = joint_att * self.op_adj_mat.unsqueeze(dim=0)  # N x O x P

        op_repr = self.emb_readout_mlp(self.op_embs.view(-1, self.op_embs.shape[2])).view_as(self.op_embs).unsqueeze(dim=0)  # 1 x O x P x F
        att_obj_repr = torch.matmul(adj_joint_att.unsqueeze(dim=2), op_repr).squeeze(dim=2)  # N x O x F
        att_pred_repr = torch.matmul(adj_joint_att.permute(0, 2, 1).unsqueeze(dim=2), op_repr.permute(0, 2, 1, 3)).squeeze(dim=2)  # N x P x F

        new_hoi_obj_logits = self.obj_output_mlp(att_obj_repr).squeeze(dim=2)
        assert new_hoi_obj_logits.shape[0] == hoi_logits.shape[0] and new_hoi_obj_logits.shape[1] == obj_logits.shape[1]
        new_hoi_logits = self.pred_output_mlp(att_pred_repr).squeeze(dim=2)
        assert new_hoi_logits.shape[0] == hoi_logits.shape[0] and new_hoi_logits.shape[1] == hoi_logits.shape[1]

        return new_hoi_obj_logits, new_hoi_logits
