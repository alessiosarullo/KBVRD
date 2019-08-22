from typing import Union

import torch
import torch.nn as nn

from lib.dataset.hico.hico_split import HicoSplit
from lib.dataset.hicodet.hicodet_split import HicoDetSplit
from lib.dataset.word_embeddings import WordEmbeddings
from lib.models.abstract_model import AbstractHOIBranch
from lib.models.misc import get_noun_verb_adj_mat


class CheatGCNBranch(AbstractHOIBranch):
    def __init__(self, dataset: Union[HicoSplit, HicoDetSplit], input_repr_dim=512, gc_dims=(256, 128), block_norm=False, **kwargs):
        super().__init__(**kwargs)
        num_gc_layers = len(gc_dims)
        self.gc_dims = gc_dims
        self.num_objects = dataset.full_dataset.num_object_classes
        self.num_actions = dataset.full_dataset.num_actions

        # Normalised adjacency matrix. Note: the identity matrix that is supposed to be added for the "renormalisation trick" (Kipf 2016) is
        # implicitly included by initialising the adjacency matrix to an identity instead of zeros.
        self.noun_verb_links = nn.Parameter(get_noun_verb_adj_mat(dataset=dataset), requires_grad=False)
        adj = torch.eye(self.num_objects + self.num_actions).float()
        if block_norm:
            # Normalised like (Kato, 2018).
            nv = self.noun_verb_links
            norm_nv = torch.diag(1 / nv.sum(dim=1).sqrt()) @ nv @ torch.diag(1 / nv.sum(dim=0).clamp(min=1).sqrt())
            adj[:self.num_objects, self.num_objects:] = norm_nv  # top right
            adj[self.num_objects:, :self.num_objects] = norm_nv.t()  # bottom left
        else:
            adj[:self.num_objects, self.num_objects:] = self.noun_verb_links  # top right
            adj[self.num_objects:, :self.num_objects] = self.noun_verb_links.t()  # bottom left
            adj = torch.diag(1 / adj.sum(dim=1).sqrt()) @ adj @ torch.diag(1 / adj.sum(dim=0).sqrt())
        self.adj = nn.Parameter(adj, requires_grad=False)

        # Starting representation
        self.z = nn.Parameter(torch.empty(self.adj.shape[0], input_repr_dim).normal_(), requires_grad=True)

        gc_layers = []
        for i in range(num_gc_layers):
            in_dim = gc_dims[i - 1] if i > 0 else input_repr_dim
            out_dim = gc_dims[i]
            if i < num_gc_layers - 1:
                gc_layers.append(nn.Sequential(nn.Linear(in_dim, out_dim),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(p=0.5)))
            else:
                gc_layers.append(nn.Linear(in_dim, out_dim))

        self.gc_layers = nn.ModuleList(gc_layers)

    @property
    def output_dim(self):
        return self.gc_dims[-1]

    def _forward(self, input_repr=None):
        if input_repr is not None:
            z = input_repr
        else:
            z = self.z
        for gcl in self.gc_layers:
            z = gcl(self.adj @ z)
        obj_embs = z[:self.num_objects]
        pred_embs = z[self.num_objects:]
        return obj_embs, pred_embs


class DetachingGCNBranch(CheatGCNBranch):
    def __init__(self, to_detach, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inds_to_detach = to_detach

    def _forward(self, input_repr=None):
        # TODO
        if input_repr is not None:
            z = input_repr
        else:
            z = self.z
        for gcl in self.gc_layers:
            aggr_z = self.adj @ z
            z = gcl(aggr_z)
        obj_embs = z[:self.num_objects]
        pred_embs = z[self.num_objects:]
        return obj_embs, pred_embs


class CheatHoiGCNBranch(AbstractHOIBranch):
    def __init__(self, dataset: Union[HicoSplit, HicoDetSplit], input_dim=512, gc_dims=(256, 128), train_z=True, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.dataset = dataset
        self.num_objects = dataset.full_dataset.num_object_classes
        self.num_actions = dataset.full_dataset.num_actions
        self.num_interactions = dataset.full_dataset.num_interactions

        self.adj = nn.Parameter(self._build_adj_matrix(), requires_grad=False)
        self.z = nn.Parameter(self._get_initial_z(), requires_grad=train_z)

        self.gc_dims = gc_dims
        self.gc_layers = nn.ModuleList(self._build_gcn())

    def _build_adj_matrix(self):
        interactions_to_obj = self.dataset.full_dataset.interaction_to_object_mat
        interactions_to_actions = self.dataset.full_dataset.interaction_to_action_mat

        # FIXME seems that normalising the whole matrix (which affects the diagonal, i.e., the contribution of the added identity) leads to poor
        #  results.

        # # This option makes this graph too sparse, isolating too much.
        # if not cfg.link_null:
        #     interactions_to_actions[:, 0] = 0

        # The adjacency matrix is:
        # | NN  NV  NA |
        # | VN  VV  VA |
        # | AN  AV  AA |
        # where N=nouns (objects), V=verbs (actions), A=actions (interactions). Since it is symmetric, NV=VN', NA=AN' and VA=AV'. Also,
        # NN=VV=AA=0 and NV=0 (connections are only present between interactions and the rest). Thus, only AN and AV need to be defined.
        # Note: the identity matrix that is supposed to be added for the "renormalisation trick" (Kipf 2016) is implicitly included by initialising
        # the adjacency matrix to an identity instead of zeros.
        adj_an = torch.from_numpy(interactions_to_obj).float()
        adj_av = torch.from_numpy(interactions_to_actions).float()
        adj_nn = torch.eye(self.num_objects).float()
        adj_vv = torch.eye(self.num_actions).float()
        adj_aa = torch.eye(self.num_interactions).float()
        zero_nv = torch.zeros((self.num_objects, self.num_actions)).float()
        adj = torch.cat([torch.cat([adj_nn,         zero_nv,        adj_an.t()], dim=1),
                         torch.cat([zero_nv.t(),    adj_vv,         adj_av.t()], dim=1),
                         torch.cat([adj_an,         adj_av,         adj_aa], dim=1)
                         ], dim=0)
        adj = torch.diag(1 / adj.sum(dim=1).sqrt()) @ adj @ torch.diag(1 / adj.sum(dim=0).sqrt())
        return adj

    def _get_initial_z(self):
        return torch.empty(self.adj.shape[0], self.input_dim).normal_()

    def _build_gcn(self):
        gc_layers = []
        num_gc_layers = len(self.gc_dims)
        for i in range(num_gc_layers):
            in_dim = self.gc_dims[i - 1] if i > 0 else self.input_dim
            out_dim = self.gc_dims[i]
            if i < num_gc_layers - 1:
                gc_layers.append(nn.Sequential(nn.Linear(in_dim, out_dim),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(p=0.5)))
            else:
                gc_layers.append(nn.Linear(in_dim, out_dim))
        return gc_layers

    @property
    def output_dim(self):
        return self.gc_dims[-1]

    def _forward(self, input_repr=None):
        if input_repr is not None:
            z = input_repr
        else:
            z = self.z
        for gcl in self.gc_layers:
            z = gcl(self.adj @ z)
        obj_embs = z[:self.num_objects]
        act_embs = z[self.num_objects:(self.num_objects + self.num_actions)]
        hoi_embs = z[(self.num_objects + self.num_actions):]
        return obj_embs, act_embs, hoi_embs


class KatoGCNBranch(CheatHoiGCNBranch):
    def __init__(self, dataset: HicoSplit, word_emb_dim, gc_dims, train_z, paper_adj, paper_gc, **kwargs):
        self.paper_adj = paper_adj
        self.paper_gc = paper_gc
        super().__init__(dataset=dataset, input_dim=word_emb_dim, gc_dims=gc_dims, train_z=train_z, **kwargs)

    def _build_adj_matrix(self):
        if not self.paper_adj:
            return super(KatoGCNBranch, self)._build_adj_matrix()

        # # # This one is not normalised properly, but it's what they use in the paper.

        def normalise(x):
            return (1 / x.sum(dim=1, keepdim=True).sqrt()) * x * (1 / x.sum(dim=0, keepdim=True).sqrt())

        interactions_to_obj = self.dataset.full_dataset.interaction_to_object_mat
        interactions_to_actions = self.dataset.full_dataset.interaction_to_action_mat

        adj_nn = normalise(torch.eye(self.num_objects).float())
        adj_vv = normalise(torch.eye(self.num_actions).float())
        adj_aa = normalise(torch.eye(self.num_interactions).float())
        adj_an = normalise(torch.from_numpy(interactions_to_obj).float())
        adj_av = normalise(torch.from_numpy(interactions_to_actions).float())
        zero_nv = torch.zeros((self.num_objects, self.num_actions)).float()
        adj = torch.cat([torch.cat([adj_nn,         zero_nv,        adj_an.t()], dim=1),
                         torch.cat([zero_nv.t(),    adj_vv,         adj_av.t()], dim=1),
                         torch.cat([adj_an,         adj_av,         adj_aa], dim=1)
                         ], dim=0)
        return adj

    def _get_initial_z(self):
        # The paper does not specify whether the embeddings are normalised or what they do for compound words.
        self.word_embs = WordEmbeddings(source='glove', dim=self.input_dim, normalize=True)
        obj_word_embs = self.word_embs.get_embeddings(self.dataset.full_dataset.objects, retry='avg')
        pred_word_embs = self.word_embs.get_embeddings(self.dataset.full_dataset.actions, retry='avg')
        return torch.cat([torch.from_numpy(obj_word_embs).float(),
                          torch.from_numpy(pred_word_embs).float(),
                          torch.zeros(self.dataset.full_dataset.num_interactions, self.input_dim)
                          ], dim=0)

    def _build_gcn(self):
        gc_layers = []
        for i in range(len(self.gc_dims)):
            in_dim = self.gc_dims[i - 1] if i > 0 else self.input_dim
            out_dim = self.gc_dims[i]
            gc_layers.append(nn.Sequential(nn.Linear(in_dim, out_dim),
                                           nn.ReLU(inplace=True)))
        return gc_layers

    def _forward(self, input_repr=None):
        assert input_repr is None

        z_n = self.z[:self.num_objects]
        z_v = self.z[self.num_objects:(self.num_objects + self.num_actions)]
        z_a = self.z[(self.num_objects + self.num_actions):]
        adj_nn = self.adj[:self.num_objects, :self.num_objects]
        adj_vv = self.adj[self.num_objects:(self.num_objects + self.num_actions), self.num_objects:(self.num_objects + self.num_actions)]
        adj_an = self.adj[(self.num_objects + self.num_actions):, :self.num_objects]
        adj_av = self.adj[(self.num_objects + self.num_actions):, self.num_objects:(self.num_objects + self.num_actions)]
        for i in range(len(self.gc_layers)):
            prev_z_n, prev_z_v, prev_z_a = z_n, z_v, z_a

            if self.paper_gc:
                # This is what they say they do. It doesn't make sense.
                z_n = self.gc_layers[i](adj_nn @ prev_z_n)
                z_v = self.gc_layers[i](adj_vv @ prev_z_v)
                z_a = self.gc_layers[i](adj_an @ prev_z_n + adj_av @ prev_z_v)
            else:
                # This is the correct way of doing it, still following their "decomposition" policy.
                z_n = self.gc_layers[i](adj_nn @ prev_z_n + adj_an.t() @ prev_z_a)
                z_v = self.gc_layers[i](adj_vv @ prev_z_v + adj_av.t() @ prev_z_a)
                z_a = self.gc_layers[i](prev_z_a + adj_an @ prev_z_n + adj_av @ prev_z_v)
        return z_n, z_v, z_a
