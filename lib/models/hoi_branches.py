import math
import numpy as np
import torch
from torch import nn as nn
from torch.nn.utils.rnn import PackedSequence

from config import cfg
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.dataset.utils import get_counts
from lib.dataset.word_embeddings import WordEmbeddings
from lib.knowledge_extractors.imsitu_knowledge_extractor import ImSituKnowledgeExtractor
from lib.knowledge_extractors.conceptnet_knowledge_extractor import ConceptnetKnowledgeExtractor
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


class KBNMotifsHOIBranch(NMotifsHOIBranch):
    def __init__(self, visual_feats_dim, obj_feat_dim, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(visual_feats_dim, obj_feat_dim, dataset, **kwargs)
        self.kb_emb_dim = 1024

        self.use_kb_sim = False
        if cfg.model.kb_sim:
            self.pred_word_embs = torch.nn.Parameter(torch.from_numpy(self.word_embs.get_embeddings(dataset.predicates)), requires_grad=True)

            w = torch.empty(dataset.num_predicates, self.kb_emb_dim)
            nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('linear'))
            self.pred_repr = torch.nn.Parameter(w, requires_grad=True)

            self.emb_fc = nn.Sequential(nn.Linear(self.kb_emb_dim + self.word_emb_dim, self.kb_emb_dim + self.word_emb_dim),
                                        nn.ReLU()
                                        )
            nn.init.xavier_normal_(self.emb_fc[0].weight, gain=nn.init.calculate_gain('relu'))

            self.post_sim = nn.Linear(self.hoi_repr_dim + self.kb_emb_dim + self.word_emb_dim, self.hoi_repr_dim)
            nn.init.xavier_normal_(self.post_sim.weight, gain=nn.init.calculate_gain('linear'))
            # torch.nn.init.normal_(self.post_sim.weight, mean=0, std=10 * math.sqrt(1.0 / self.hoi_repr_dim))
            torch.nn.init.zeros_(self.post_sim.bias)

            op_adj_mat = np.zeros([dataset.num_object_classes, dataset.num_predicates])
            if cfg.model.use_imsitu:
                imsitu_counts = ImSituKnowledgeExtractor().extract_freq_matrix(dataset)
                imsitu_counts[:, 0] = 0  # exclude null interaction
                op_adj_mat += np.minimum(1, imsitu_counts)  # only check if the pair exists (>=1 occurrence) or not (0 occurrences)
            if cfg.model.freq_bias:
                int_counts = get_counts(dataset=dataset)
                int_counts[:, 0] = 0  # exclude null interaction
                op_adj_mat += np.minimum(1, int_counts)  # only check if the pair exists (>=1 occurrence) or not (0 occurrences)
            op_adj_mat = np.minimum(1, op_adj_mat)  # does not matter if the same information is present in different sources TODO try without?

            if np.any(op_adj_mat):
                op_adj_mat /= np.maximum(1, np.sum(op_adj_mat, axis=1, keepdims=True))  # normalise
                self.op_adj_mat = torch.nn.Parameter(torch.from_numpy(op_adj_mat).float(), requires_grad=False)  # TODO check if training this helps
                self.use_kb_sim = True
        print('KB sim:', self.use_kb_sim)

    def _forward(self, boxes_ext, box_repr, union_boxes_feats, hoi_infos, box_labels=None):
        hoi_repr = super()._forward(boxes_ext, box_repr, union_boxes_feats, hoi_infos, box_labels)

        if self.use_kb_sim:
            obj_gc_repr = self.op_adj_mat @ self.emb_fc(torch.cat([self.pred_word_embs, self.pred_repr], dim=1))
            if box_labels is not None:
                hoi_obj_classes = box_labels[hoi_infos[:, 2]].detach()
                hoi_repr = self.post_sim(torch.cat([hoi_repr, obj_gc_repr[hoi_obj_classes, :]], dim=1))
            else:
                hoi_obj_classes = boxes_ext[hoi_infos[:, 2], 5:].detach()
                attended_obj_gc_repr = hoi_obj_classes @ obj_gc_repr
                hoi_repr = self.post_sim(torch.cat([hoi_repr, attended_obj_gc_repr], dim=1))

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


class KBHoiBranch(AbstractHOIBranch):
    def __init__(self, visual_feats_dim, obj_repr_dim, dataset: HicoDetInstanceSplit, **kwargs):
        # TODO docs and FIXME comments
        super().__init__(**kwargs)
        self.num_objects = dataset.num_object_classes
        self.hoi_repr_dim = visual_feats_dim

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
            cnet_counts = ConceptnetKnowledgeExtractor().extract_freq_matrix(dataset=dataset)
            cnet_counts[:, 0] = 0  # exclude null interaction
            op_adj_mats.append(np.minimum(1, cnet_counts))  # only check if the pair exists (>=1 occurrence) or not (0 occurrences)

        assert op_adj_mats
        op_adj_mats = np.stack(op_adj_mats, axis=2)[:, :, :, None]  # O x P x S x 1
        self.op_adj_mat = torch.nn.Parameter(torch.from_numpy(op_adj_mats).float(), requires_grad=False)
        self.op_conf_mat = torch.nn.Parameter(torch.from_numpy(op_adj_mats).float(), requires_grad=True)

        op_repr = torch.empty(list(op_adj_mats.shape[:-1]) + [self.hoi_repr_dim])  # O x P x S x F
        nn.init.xavier_normal_(op_repr, gain=1.0)
        self.op_repr = torch.nn.Parameter(op_repr, requires_grad=True)
        # print(op_adj_mats.shape, op_repr.shape)

        self.src_att = nn.Sequential(nn.Linear(visual_feats_dim, op_adj_mats.shape[2]),
                                     nn.Softmax(dim=1))
        self.pred_att_fc = nn.Sequential(nn.Linear(visual_feats_dim, dataset.num_predicates),
                                         nn.Softmax(dim=1))

        self.union_repr_fc = nn.Linear(visual_feats_dim, self.hoi_repr_dim)
        nn.init.xavier_normal_(self.union_repr_fc.weight, gain=1.0)

        self.hoi_obj_repr_fc = nn.Linear(obj_repr_dim, self.hoi_repr_dim)
        nn.init.xavier_normal_(self.hoi_obj_repr_fc.weight, gain=1.0)

    def _forward(self, boxes_ext, obj_repr, union_boxes_feats, hoi_infos, obj_logits, box_labels=None):
        if box_labels is not None:
            box_predict = obj_repr.new_zeros((box_labels.shape[0], self.num_objects))
            box_predict[torch.arange(box_predict.shape[0]), box_labels] = 1
        else:
            box_predict = nn.functional.softmax(obj_logits, dim=1)
        obj_att = box_predict[hoi_infos[:, 2], :]  # N x O
        pred_att = self.pred_att_fc(union_boxes_feats)  # N x P
        src_att = self.src_att(union_boxes_feats)  # N x S

        batch_size = union_boxes_feats.shape[0]
        att = obj_att.view(batch_size, -1, 1, 1) * pred_att.view(batch_size, 1, -1, 1) * src_att.view(batch_size, 1, 1, -1)  # N x O x P x S

        ext_op_repr = self.op_adj_mat * torch.sigmoid(self.op_conf_mat) * self.op_repr  # O x P x S x F
        hoi_ext_repr = att.view(batch_size, -1) @ ext_op_repr.view(-1, ext_op_repr.shape[-1])  # N x F

        hoi_obj_repr = obj_repr[hoi_infos[:, 2], :]
        hoi_repr = union_boxes_feats + self.hoi_obj_repr_fc(hoi_obj_repr) + hoi_ext_repr

        return hoi_repr


class Mem2HoiBranch(AbstractHOIBranch):
    def __init__(self, visual_feats_dim, obj_repr_dim, dataset, **kwargs):
        self.cells_per_type = 3
        self.sim_thr = 0.6  # threshold over cosine similarity
        super().__init__(**kwargs)
        self.memory_repr_dim = visual_feats_dim

        # HOI object repr
        self.hoi_obj_repr_fc = nn.Linear(obj_repr_dim, self.memory_repr_dim)
        torch.nn.init.xavier_normal_(self.hoi_obj_repr_fc.weight, gain=1.0)

        # Memory
        memory_keys = torch.zeros((dataset.num_predicates, self.cells_per_type, self.memory_repr_dim))
        self.memory_keys = torch.nn.Parameter(memory_keys, requires_grad=False)

        memory_values = torch.zeros(memory_keys.shape)
        self.memory_values = torch.nn.Parameter(memory_values, requires_grad=False)

        self.cell_correlations = torch.zeros((memory_keys.shape[0], memory_keys.shape[1], memory_keys.shape[1]))

    @property
    def output_dim(self):
        return self.memory_repr_dim

    def _forward(self, boxes_ext, box_repr, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        # Memory is M x N x F

        ubf_norm = torch.nn.functional.normalize(union_boxes_feats)
        arange = torch.arange(0, ubf_norm.shape[0])
        mem_sim = (ubf_norm[:, None, None, :] * self.memory_keys[None, :, :, :]).sum(dim=-1)  # H x M x N

        best_type_per_hoi = mem_sim.max(dim=2)[0].argmax(dim=1)
        best_cell_per_hoi = mem_sim[arange, best_type_per_hoi, :].argmax(dim=-1)

        sim_per_hoi = mem_sim[arange, best_type_per_hoi, best_cell_per_hoi]
        mem_hits = (sim_per_hoi >= self.sim_thr)

        hoi_repr = torch.empty_like(union_boxes_feats)
        hoi_repr[mem_hits, :] = self.memory_values[best_type_per_hoi[mem_hits], best_cell_per_hoi[mem_hits], :].detach()
        if not mem_hits.all():
            hoi_repr[~mem_hits, :] = self.hoi_obj_repr_fc(box_repr[hoi_infos[~mem_hits, 2], :]) + union_boxes_feats[~mem_hits, :]

        if hoi_labels is not None:
            correlations = (mem_sim[:, :, :, None] * mem_sim[:, :, None, :]).mean(dim=0)
            self.cell_correlations = (1 - 0.1) * self.cell_correlations + 0.1 * correlations.cpu()  # FIXME magic constant, cpu()

            cells_to_update = self.cell_correlations.sum(dim=2).argmax(dim=1)  # update cell with max correlation

            incorrect_hoi_inds = (1 - hoi_labels[arange, best_type_per_hoi]).byte()
            incorrect_hoi_labels_t = hoi_labels[incorrect_hoi_inds, :].t()

            preds_to_update = incorrect_hoi_labels_t.byte().any(dim=1)
            incorrect_hoi_labels_t = incorrect_hoi_labels_t[preds_to_update, :]
            num_ex_per_preds = incorrect_hoi_labels_t.sum(dim=1, keepdim=True).clamp(min=1)

            key_update_vec = incorrect_hoi_labels_t @ ubf_norm[incorrect_hoi_inds, :] / num_ex_per_preds
            self.memory_keys[preds_to_update, cells_to_update[preds_to_update], :] = key_update_vec
            value_update_vec = incorrect_hoi_labels_t @ hoi_repr[incorrect_hoi_inds, :] / num_ex_per_preds
            self.memory_values[preds_to_update, cells_to_update[preds_to_update], :] = value_update_vec

        return hoi_repr


class MemHoiBranch(AbstractHOIBranch):
    def __init__(self, visual_feats_dim, obj_repr_dim, dataset, **kwargs):
        self.word_emb_dim = 300
        super().__init__(**kwargs)
        self.memory_input_size = visual_feats_dim
        self.memory_repr_size = self.word_emb_dim
        self.memory_output_size = visual_feats_dim

        # HOI object repr
        self.hoi_obj_repr_fc = nn.Linear(obj_repr_dim, visual_feats_dim)
        torch.nn.init.xavier_normal_(self.hoi_obj_repr_fc.weight, gain=1.0)

        # Memory
        word_embs = WordEmbeddings(source='glove', dim=self.word_emb_dim)
        pred_word_embs = torch.from_numpy(word_embs.get_embeddings(dataset.predicates))
        memory_keys = torch.nn.functional.normalize(pred_word_embs)

        self.memory_mapping_fc = nn.Linear(visual_feats_dim, memory_keys.shape[1])
        torch.nn.init.xavier_normal_(self.memory_mapping_fc.weight, gain=1.0)

        memory_values = torch.empty((dataset.num_predicates, visual_feats_dim))
        torch.nn.init.xavier_normal_(memory_values, gain=1.0)

        # self.memory_att_temp_fc = nn.Linear(visual_feats_dim, 1)

        self.memory_keys = torch.nn.Parameter(memory_keys, requires_grad=False)
        self.memory_values = torch.nn.Parameter(memory_values, requires_grad=True)

        # self.memory_readout_fc = nn.Sequential(nn.Linear(self.memory_repr_size, self.memory_output_size),
        #                                        nn.ReLU())
        # torch.nn.init.xavier_normal_(self.memory_readout_fc[0].weight, gain=nn.init.calculate_gain('relu'))

    @property
    def output_dim(self):
        return self.memory_output_size

    def _forward(self, boxes_ext, box_repr, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        # Memory is M x F

        hoi_repr = self.hoi_obj_repr_fc(box_repr[hoi_infos[:, 2], :]) + union_boxes_feats
        mem_hoi_repr = hoi_repr.detach()  # H x F

        input_mem_repr = self.memory_mapping_fc(mem_hoi_repr)
        input_mem_repr = torch.nn.functional.normalize(input_mem_repr)
        # input_mem_repr = mem_hoi_repr

        mem_sim = input_mem_repr @ self.memory_keys.t()  # H x M
        # mem_att_temp = self.memory_att_temp_fc(mem_hoi_repr)
        mem_att_temp = 1
        mem_att = torch.nn.functional.softmax(mem_att_temp * mem_sim, dim=1)  # H x M

        mem_repr = mem_att @ self.memory_values  # H x F

        # mem_output = self.memory_readout_fc(mem_repr)
        mem_output = mem_repr

        hoi_repr = hoi_repr + mem_output

        if hoi_labels is not None:
            mem_pred = hoi_labels @ self.memory_values

            mem_sim_t = (input_mem_repr[:, None, :] * self.memory_keys[None, :, :]).sum(dim=2)  # H x M
            margin_loss = ((1 - hoi_labels) * mem_sim_t - hoi_labels * mem_sim_t).mean(dim=1)
        else:
            mem_pred = margin_loss = None

        return hoi_repr, mem_pred, margin_loss


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
        self.interactions_to_obj = nn.Parameter(torch.from_numpy(interactions_to_obj), requires_grad=False)
        self.interactions_to_preds = nn.Parameter(torch.from_numpy(interactions_to_preds), requires_grad=False)
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

        op_embs = torch.cat([obj_embs[hoi_infos[:, 2], :], pred_embs], dim=1)
        op_sims = self.op_cossim(op_embs.unsqueeze(dim=2), self.interaction_embs.unsqueeze(dim=0))

        hoi_obj_logits = op_sims @ self.interactions_to_obj
        hoi_logits = op_sims @ self.interactions_to_preds

        return hoi_logits, hoi_obj_logits


class HoiMemGCBranch(AbstractHOIBranch):
    def __init__(self, visual_feats_dim, dataset: HicoDetInstanceSplit, **kwargs):
        # TODO docs and FIXME comments
        self.hoi_repr_dim = 600
        super().__init__(**kwargs)
        self.num_objects = dataset.num_object_classes

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
            cnet_counts = ConceptnetKnowledgeExtractor().extract_freq_matrix(dataset=dataset)
            cnet_counts[:, 0] = 0  # exclude null interaction
            op_adj_mats.append(np.minimum(1, cnet_counts))  # only check if the pair exists (>=1 occurrence) or not (0 occurrences)

        assert op_adj_mats
        op_adj_mats = np.stack(op_adj_mats, axis=2)  # O x P x S
        op_adj_mat = np.sum(op_adj_mats, axis=2, keepdims=True)  # O x P x 1
        self.op_adj_mat = torch.nn.Parameter(torch.from_numpy(op_adj_mat).float(), requires_grad=True)

        op_feats = torch.empty((dataset.num_object_classes, dataset.num_predicates, visual_feats_dim))
        nn.init.xavier_uniform_(op_feats, gain=1.0)
        op_pairs = torch.zeros((dataset.num_object_classes, dataset.num_predicates))
        for i in range(dataset.num_images):
            ex = dataset.get_entry(i)
            hoi_l, obj_l = ex.precomp_hoi_labels, ex.precomp_box_labels
            union_feats, infos = ex.precomp_hoi_union_feats, ex.precomp_hoi_infos
            hoi_obj_l = obj_l[infos[:, 2]]
            assert union_feats.shape[0] == hoi_l.shape[0] == hoi_obj_l.shape[0]
            for ol, hl, uf in zip(hoi_obj_l, hoi_l, union_feats):
                hl = np.flatnonzero(hl)
                op_feats[ol, hl, :] += torch.from_numpy(uf.astype(np.float32, copy=False))
                op_pairs[ol, hl] += 1
        self.op_repr = torch.nn.Parameter(op_feats / op_pairs.clamp(min=1).unsqueeze(dim=2), requires_grad=True)
        assert self.op_repr.shape[0] == self.op_adj_mat.shape[0], (self.op_repr.shape, self.op_adj_mat.shape)
        assert self.op_repr.shape[1] == self.op_adj_mat.shape[1], (self.op_repr.shape, self.op_adj_mat.shape)

        self.hoi_repr_fc = nn.Linear(visual_feats_dim, self.hoi_repr_dim)
        nn.init.xavier_normal_(self.hoi_repr_fc.weight, gain=1.0)
        self.obj_readout_fc = nn.Linear(op_feats.shape[2], self.hoi_repr_dim)
        nn.init.xavier_normal_(self.obj_readout_fc.weight, gain=1.0)
        self.pred_readout_fc = nn.Linear(op_feats.shape[2], self.hoi_repr_dim)
        nn.init.xavier_normal_(self.pred_readout_fc.weight, gain=1.0)

    @property
    def output_dim(self):
        return self.hoi_repr_dim

    def _forward(self, hoi_logits, obj_logits, union_box_feats, hoi_infos, box_labels=None, hoi_labels=None):
        if box_labels is not None:
            assert hoi_labels is not None
            obj_prediction = union_box_feats.new_zeros((box_labels.shape[0], self.num_objects))
            obj_prediction[torch.arange(obj_prediction.shape[0]), box_labels] = 1
            hoi_prediction = hoi_labels
        else:
            assert hoi_labels is None
            obj_prediction = nn.functional.softmax(obj_logits, dim=1)
            hoi_prediction = torch.sigmoid(hoi_logits)

        obj_att = obj_prediction[hoi_infos[:, 2], :]  # N x O
        pred_att = hoi_prediction  # N x P

        op_repr = (self.op_adj_mat * self.op_repr)  # O x P x F
        ext_obj_repr = self.obj_readout_fc(obj_att @ op_repr.mean(dim=1))
        ext_pred_repr = self.pred_readout_fc(pred_att @ op_repr.mean(dim=0))
        union_repr = self.hoi_repr_fc(union_box_feats)

        hoi_repr = union_repr + ext_obj_repr + ext_pred_repr

        return hoi_repr
