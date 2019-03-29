import math

import numpy as np
import torch
import torch.nn as nn

from config import cfg
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.dataset.utils import get_counts
from lib.dataset.word_embeddings import WordEmbeddings
from lib.knowledge_extractors.imsitu_knowledge_extractor import ImSituKnowledgeExtractor
from lib.models.abstract_model import AbstractHOIBranch
from lib.models.context_modules import SpatialContext, ObjectContext
from lib.models.generic_model import GenericModel
from lib.models.highway_lstm_cuda.alternating_highway_lstm import AlternatingHighwayLSTM
from lib.models.nmotifs.lincontext import sort_rois, PackedSequence


class KBModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'kb'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        self.imsitu_prior_emb_dim = 256
        self.imsitu_branch_final_emb_dim = 512
        super().__init__(dataset, **kwargs)

        self.spatial_context_branch = SpatialContext(input_dim=2 * (self.visual_module.mask_resolution ** 2))
        self.obj_branch = ObjectContext(input_dim=self.visual_module.vis_feat_dim +
                                                  self.dataset.num_object_classes +
                                                  self.spatial_context_branch.output_dim)
        if cfg.model.use_memory:
            self.hoi_branch = MemNMotifsHOIBranch(self.visual_module.vis_feat_dim, self.obj_branch.output_repr_dim, dataset)
        else:
            self.hoi_branch = KBNMotifsHOIBranch(self.visual_module.vis_feat_dim, self.obj_branch.output_repr_dim, dataset)

        self.obj_output_fc = nn.Linear(self.obj_branch.output_repr_dim, self.dataset.num_object_classes)
        self.hoi_output_fc = nn.Linear(self.hoi_branch.output_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

        self.hoi_refinement_branch = KBHOIBiasBranch(dataset, self.hoi_branch.output_dim)

        self.values_to_monitor = {}  # FIXME delete

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        spatial_ctx, spatial_rels_feats = self.spatial_context_branch(masks, im_ids, hoi_infos)
        obj_ctx, obj_repr = self.obj_branch(boxes_ext, box_feats, spatial_ctx, im_ids, box_im_ids)
        hoi_repr = self.hoi_branch(boxes_ext, obj_repr, union_boxes_feats, hoi_infos, box_labels)

        obj_logits = self.obj_output_fc(obj_repr)
        hoi_logits = self.hoi_output_fc(hoi_repr)

        hoi_logits = self.hoi_refinement_branch(hoi_logits, hoi_repr, boxes_ext, hoi_infos, box_labels)

        for k, v in self.hoi_refinement_branch.values_to_monitor.items():  # FIXME delete
            self.values_to_monitor[k] = v

        return obj_logits, hoi_logits


class NMotifsHOIBranch(AbstractHOIBranch):
    def __init__(self, visual_feats_dim, obj_feat_dim, dataset, **kwargs):
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
        perm, inv_perm, ls_transposed = sort_rois(self.order, box_im_ids, box_priors=boxes_ext[:, 1:5])
        hoi_output_obj_repr = self.hoi_obj_birnn(PackedSequence(hoi_input_obj_repr[perm], ls_transposed))[0]
        hoi_output_obj_repr = hoi_output_obj_repr[inv_perm]

        hoi_ho_repr = self.post_lstm(hoi_output_obj_repr)
        hoi_ho_repr = hoi_ho_repr.view(hoi_ho_repr.shape[0], 2, -1)  # Split into subject and object representations
        subj_repr = hoi_ho_repr[:, 0][subj_inds]
        dobj_repr = hoi_ho_repr[:, 1][dobj_inds]
        hoi_repr = subj_repr * dobj_repr * union_boxes_feats

        return hoi_repr


class MemNMotifsHOIBranch(NMotifsHOIBranch):
    def __init__(self, visual_feats_dim, obj_feat_dim, dataset, **kwargs):
        self.mem_att_entropy = 1 / 0.1
        super().__init__(visual_feats_dim, obj_feat_dim, dataset, **kwargs)
        self.memory_input_size = self.hoi_repr_dim
        self.memory_output_size = self.hoi_repr_dim
        memory_size = dataset.num_predicates  # this CANNOT be modified without changing the forward pass

        mem = torch.empty(memory_size, self.memory_input_size)
        nn.init.xavier_uniform_(mem, gain=1.0)
        self.memory_keys = torch.nn.Parameter(torch.nn.functional.normalize(mem), requires_grad=True)
        self.memory_attention = nn.Sequential(nn.Linear(self.hoi_repr_dim, 1),
                                              nn.Sigmoid())
        self.memory_readout_fc = nn.Sequential(nn.Linear(self.memory_input_size, self.memory_output_size),
                                               nn.ReLU())

    def _forward(self, boxes_ext, box_repr, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        hoi_repr = super()._forward(boxes_ext, box_repr, union_boxes_feats, hoi_infos, box_labels)

        hor_repr_norm = torch.nn.functional.normalize(hoi_repr)
        memory_sim = hor_repr_norm @ self.memory_keys.t()
        memory_att = torch.nn.functional.softmax(self.mem_att_entropy * memory_sim, dim=1)
        memory_repr = torch.nn.functional.normalize(memory_att @ self.memory_keys)
        memory_output = self.memory_readout_fc(memory_repr)
        hoi_repr += memory_output

        if hoi_labels is not None:
            misses = 1 - hoi_labels[torch.arange(hoi_labels.shape[0]), memory_att.argmax(dim=1)]
            updates = hoi_labels.t() @ (misses.view(-1, 1) * memory_repr)
            self.memory_keys = torch.nn.normalize(self.memory_keys + updates.deatch())

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
                imsitu_counts = ImSituKnowledgeExtractor().extract_prior_matrix(dataset)
                imsitu_counts[:, 0] = 0  # exclude null interaction
                op_adj_mat += np.minimum(1, imsitu_counts)  # only check if the pair exists (>=1 occurrence) or not (0 occurrences)
            if cfg.model.use_int_freq:
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


class KBHOIBiasBranch(AbstractHOIBranch):
    def __init__(self, dataset: HicoDetInstanceSplit, hoi_repr_dim, **kwargs):
        self.gc_repr_dim = 256
        super().__init__(**kwargs)

        # Freq bias
        freqs = []
        if cfg.model.use_int_freq:
            int_counts = get_counts(dataset=dataset)
            freqs.append(int_counts)

        if freqs:
            self.bias_priors = nn.ModuleList()
            for fmat in freqs:
                priors = np.log(fmat / np.maximum(1, np.sum(fmat, axis=1, keepdims=True)) + 1e-3)  # FIXME magic constant
                self.bias_priors.append(torch.nn.Embedding.from_pretrained(torch.from_numpy(priors).float(), freeze=not cfg.model.train_prior))

            if cfg.model.prior_att:
                self.prior_source_attention = nn.Sequential(nn.Linear(hoi_repr_dim, len(self.bias_priors)),
                                                            nn.Sigmoid())
            else:
                self.prior_source_attention = None
        else:
            self.bias_priors = None

    def _forward(self, hoi_logits, hoi_repr, boxes_ext, hoi_infos, box_labels=None):

        obj_classes = box_labels if box_labels is not None else torch.argmax(boxes_ext[:, 5:], dim=1)
        hoi_obj_classes = obj_classes[hoi_infos[:, 2]].detach()

        if self.bias_priors is not None:
            priors = torch.stack([prior(hoi_obj_classes) for prior in self.bias_priors], dim=0).exp()

            if self.prior_source_attention is not None:
                src_att = self.prior_source_attention(hoi_repr)
                prior_contribution = (src_att.t().unsqueeze(dim=2) * priors).sum(dim=0)
                self.values_to_monitor['hoi_attention'] = src_att.detach().cpu().numpy()
            else:
                prior_contribution = priors.sum(dim=0)
            hoi_logits += prior_contribution.log()

        return hoi_logits
