from typing import Union

import torch
import torch.nn as nn

from config import cfg
from lib.dataset.hico.hico_split import HicoSplit
from lib.dataset.hicodet.hicodet_split import HicoDetSplit
from lib.dataset.word_embeddings import WordEmbeddings
from lib.models.abstract_model import AbstractHOIBranch
from lib.models.misc import get_noun_verb_adj_mat


class CheatGCNBranch(AbstractHOIBranch):
    def __init__(self, dataset: Union[HicoSplit, HicoDetSplit], input_repr_dim=512, gc_dims=(256, 128), **kwargs):
        super().__init__(**kwargs)
        num_gc_layers = len(gc_dims)
        self.gc_dims = gc_dims
        self.num_objects = dataset.full_dataset.num_object_classes
        self.num_actions = dataset.full_dataset.num_actions

        # Normalised adjacency matrix. Note: the identity matrix that is supposed to be added for the "renormalisation trick" (Kipf 2016) is
        # implicitly included by initialising the adjacency matrix to an identity instead of zeros.
        self.noun_verb_links = nn.Parameter(get_noun_verb_adj_mat(dataset=dataset), requires_grad=False)
        adj = torch.eye(self.num_objects + self.num_actions).float()
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


class CheatHoiGCNBranch(AbstractHOIBranch):
    def __init__(self, dataset: Union[HicoSplit, HicoDetSplit], input_repr_dim=512, gc_dims=(256, 128), **kwargs):
        super().__init__(**kwargs)
        self.num_objects = dataset.full_dataset.num_object_classes
        self.num_actions = dataset.full_dataset.num_actions
        self.num_interactions = dataset.full_dataset.num_interactions

        interactions_to_obj = dataset.full_dataset.interaction_to_object_mat
        interactions_to_actions = dataset.full_dataset.interaction_to_action_mat
        if not cfg.link_null:
            interactions_to_actions[:, 0] = 0

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
        adj = torch.eye(self.num_objects + self.num_actions + self.num_interactions).float()
        adj[0:self.num_objects, (self.num_objects + self.num_actions):] = adj_an.t()
        adj[self.num_objects:(self.num_objects + self.num_actions), (self.num_objects + self.num_actions):] = adj_av.t()
        adj[(self.num_objects + self.num_actions):, 0:self.num_objects] = adj_an
        adj[(self.num_objects + self.num_actions):, self.num_objects:(self.num_objects + self.num_actions)] = adj_av
        adj = torch.diag(1 / adj.sum(dim=1).sqrt()) @ adj @ torch.diag(1 / adj.sum(dim=0).sqrt())
        self.adj = nn.Parameter(adj, requires_grad=False)

        self.z = nn.Parameter(torch.empty(self.adj.shape[0], input_repr_dim).normal_(), requires_grad=True)

        self.gc_dims = gc_dims
        num_gc_layers = len(self.gc_dims)
        self.gc_layers = nn.ModuleList()
        for i in range(num_gc_layers):
            in_dim = gc_dims[i - 1] if i > 0 else input_repr_dim
            out_dim = gc_dims[i]
            if i < num_gc_layers - 1:
                self.gc_layers.append(nn.Sequential(nn.Linear(in_dim, out_dim),
                                                    nn.ReLU(inplace=True),
                                                    nn.Dropout(p=0.5)))
            else:
                self.gc_layers.append(nn.Linear(in_dim, out_dim))

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
    def __init__(self, dataset: HicoSplit, word_emb_dim=200, gc_dims=(512, 200), train_z=True, **kwargs):
        super().__init__(dataset=dataset, input_repr_dim=word_emb_dim, gc_dims=gc_dims, **kwargs)

        # Note that the adjacency matrix used in the convolution will have a different form than the one written in the paper, because that one
        # doesn't make any sense.

        self.word_embs = WordEmbeddings(source='glove', dim=word_emb_dim, normalize=True)
        obj_word_embs = self.word_embs.get_embeddings(dataset.full_dataset.objects, retry='avg')
        pred_word_embs = self.word_embs.get_embeddings(dataset.full_dataset.actions, retry='avg')

        self.z = nn.Parameter(torch.cat([torch.from_numpy(obj_word_embs).float(),
                                         torch.from_numpy(pred_word_embs).float(),
                                         torch.zeros(dataset.full_dataset.num_interactions, word_emb_dim)
                                         ], dim=0),
                              requires_grad=train_z)

        self.gc_layers = []
        for i in range(len(self.gc_dims)):
            in_dim = gc_dims[i - 1] if i > 0 else word_emb_dim
            out_dim = gc_dims[i]
            self.gc_layers.append(nn.Sequential(nn.Linear(in_dim, out_dim),
                                                nn.ReLU(inplace=True)))
