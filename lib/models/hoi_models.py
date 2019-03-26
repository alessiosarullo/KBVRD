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
        self.hoi_branch = NMotifsHOIBranch(self.visual_module.vis_feat_dim, self.obj_branch.output_repr_dim, dataset.objects)

        self.obj_output_fc = nn.Linear(self.obj_branch.output_repr_dim, self.dataset.num_object_classes)
        if cfg.model.lsig:
            self.hoi_output_fc = [nn.Linear(self.hoi_branch.output_dim, dataset.num_predicates, bias=True),
                                  nn.LogSigmoid()]
        else:
            self.hoi_output_fc = nn.Linear(self.hoi_branch.output_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

        freqs = []
        if cfg.model.use_imsitu:
            imsitu_counts = ImSituKnowledgeExtractor().extract_prior_matrix(self.dataset)
            freqs.append(imsitu_counts)
        if cfg.model.use_int_freq:
            int_counts = get_counts(dataset=dataset)
            freqs.append(int_counts)

        if freqs:
            self.priors = nn.ModuleList()
            for fmat in freqs:
                priors = np.log(fmat / np.maximum(1, np.sum(fmat, axis=1, keepdims=True)) + 1e-3)  # FIXME magic constant
                self.priors.append(torch.nn.Embedding.from_pretrained(torch.from_numpy(priors).float(), freeze=False))

            if cfg.model.prior_att:
                self.prior_source_attention = nn.Sequential(nn.Linear(self.hoi_branch.output_dim, len(self.priors)),
                                                            nn.Sigmoid())
            else:
                self.prior_source_attention = None

            # self.prior_weight = nn.Sequential(nn.Linear(self.hoi_branch.output_dim, 1),
            #                                   nn.Sigmoid())
            # torch.nn.init.xavier_uniform_(self.prior_weight[0].weight, gain=torch.nn.init.calculate_gain('sigmoid'))
        else:
            self.priors = None

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

        if self.priors is not None:
            obj_classes = torch.argmax(boxes_ext[:, 5:], dim=1)  # FIXME in nmotifs they use actual labels during training
            hoi_obj_classes = obj_classes[hoi_infos[:, 2]]
            priors = torch.stack([prior(hoi_obj_classes) for prior in self.priors], dim=0)

            if self.prior_source_attention is not None:
                src_att = self.prior_source_attention(hoi_repr)
                hoi_logits = hoi_logits + (src_att.t().unsqueeze(dim=2) * priors).sum(dim=0)
            else:
                hoi_logits = hoi_logits + priors.sum(dim=0)

        return obj_logits, hoi_logits


class NMotifsHOIBranch(AbstractHOIBranch):
    def __init__(self, visual_feats_dim, obj_feat_dim, objects_class_names, **kwargs):
        self.word_emb_dim = 200
        self.edge_ctx_num_layers = 4
        self.rnn_hidden_dim = 256
        self.order = 'leftright'
        self.dropout_rate = 0.1
        super().__init__(**kwargs)
        self.hoi_repr_dim = visual_feats_dim

        word_embeddings = WordEmbeddings(source='glove', dim=self.word_emb_dim).get_embeddings(objects_class_names)
        self.word_embs = torch.nn.Embedding.from_pretrained(torch.from_numpy(word_embeddings), freeze=True)

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
        hoi_input_obj_repr = torch.cat((self.word_embs(obj_classes), box_repr), dim=1)
        perm, inv_perm, ls_transposed = sort_rois(self.order, box_im_ids, box_priors=boxes_ext[:, 1:5])
        hoi_output_obj_repr = self.hoi_obj_birnn(PackedSequence(hoi_input_obj_repr[perm], ls_transposed))[0]
        hoi_output_obj_repr = hoi_output_obj_repr[inv_perm]

        hoi_ho_repr = self.post_lstm(hoi_output_obj_repr)
        hoi_ho_repr = hoi_ho_repr.view(hoi_ho_repr.shape[0], 2, -1)  # Split into subject and object representations
        subj_repr = hoi_ho_repr[:, 0][subj_inds]
        dobj_repr = hoi_ho_repr[:, 1][dobj_inds]
        hoi_repr = subj_repr * dobj_repr * union_boxes_feats

        return hoi_repr
