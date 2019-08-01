import numpy as np
import torch
import torch.nn as nn

from lib.dataset.hicodet.hicodet_split import HicoDetSplit
from lib.dataset.hico.hico_split import HicoSplit
from lib.dataset.word_embeddings import WordEmbeddings
from lib.models.abstract_model import AbstractHOIBranch
from lib.models.misc import get_noun_verb_adj_mat


class CheatGCNBranch(AbstractHOIBranch):
    def __init__(self, dataset: HicoDetSplit, input_repr_dim=512, gc_dims=(256, 128), **kwargs):
        super().__init__(**kwargs)
        num_gc_layers = len(gc_dims)
        self.gc_dims = gc_dims
        self.num_objects = dataset.full_dataset.num_object_classes
        self.num_actions = dataset.full_dataset.num_actions

        # Normalised adjacency matrix
        self.noun_verb_links = nn.Parameter(get_noun_verb_adj_mat(dataset=dataset), requires_grad=False)
        adj = torch.eye(self.num_objects + self.num_actions).float()
        adj[:self.num_objects, self.num_objects:] = self.noun_verb_links  # top right
        adj[self.num_objects:, :self.num_objects] = self.noun_verb_links.t()  # bottom left
        adj = torch.diag(1 / adj.sum(dim=1).sqrt()) @ adj @ torch.diag(1 / adj.sum(dim=0).sqrt())

        self.adj = nn.Parameter(adj, requires_grad=False)
        self.adj_nv = nn.Parameter(adj[:self.num_objects, self.num_objects:], requires_grad=False)
        self.adj_diag = nn.Parameter(adj.diag(), requires_grad=False)

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
        raise self.gc_dims[-1]

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


class ExtCheatGCNBranch(AbstractHOIBranch):
    def __init__(self, dataset: HicoDetSplit, input_repr_dim=512, gc_dims=(256, 128), **kwargs):
        super().__init__(**kwargs)
        num_gc_layers = len(gc_dims)
        self.gc_dims = gc_dims
        self.num_objects = dataset.full_dataset.num_object_classes
        self.num_actions = dataset.full_dataset.num_actions

        self.word_embs = WordEmbeddings(source='glove', dim=300, normalize=True)
        pred_word_embs = self.word_embs.get_embeddings(dataset.full_dataset.predicates, retry='avg')
        pred_emb_sim = pred_word_embs @ pred_word_embs.T
        pred_emb_sim[np.arange(pred_emb_sim.shape[0]), np.arange(pred_emb_sim.shape[0])] = 1

        # Normalised adjacency matrix
        self.noun_verb_links = nn.Parameter(torch.from_numpy((dataset.full_dataset.op_pair_to_interaction >= 0).astype(np.float32)), requires_grad=False)
        adj = torch.eye(self.num_objects + self.num_actions).float()
        adj[:self.num_objects, self.num_objects:] = self.noun_verb_links  # top right
        adj[self.num_objects:, :self.num_objects] = self.noun_verb_links.t()  # bottom left
        adj[self.num_objects:, self.num_objects:] = torch.from_numpy(pred_emb_sim)  # bottom right
        adj = torch.diag(1 / adj.sum(dim=1).sqrt()) @ adj @ torch.diag(1 / adj.sum(dim=0).sqrt())

        self.adj = nn.Parameter(adj, requires_grad=False)
        self.adj_nv = nn.Parameter(adj[:self.num_objects, self.num_objects:], requires_grad=False)
        self.adj_vv = nn.Parameter(adj[self.num_objects:, self.num_objects:], requires_grad=False)
        self.adj_diag = nn.Parameter(adj.diag(), requires_grad=False)

        # Starting representation
        self.z = nn.Parameter(torch.empty(self.adj.shape[0], input_repr_dim).normal_(),
                              requires_grad=True)

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
        raise self.gc_dims[-1]

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
    def __init__(self, dataset: HicoDetSplit, input_repr_dim=512, gc_dims=(256, 128), **kwargs):
        super().__init__(**kwargs)
        num_gc_layers = len(gc_dims)
        self.gc_dims = gc_dims
        self.num_objects = dataset.full_dataset.num_object_classes
        self.num_actions = dataset.full_dataset.num_actions
        self.num_interactions = dataset.full_dataset.num_interactions

        interactions = dataset.full_dataset.interactions  # each is [p, o]
        assert self.num_interactions == interactions.shape[0] == 600
        interactions_to_obj = np.zeros((self.num_interactions, self.num_objects))
        interactions_to_obj[np.arange(self.num_interactions), interactions[:, 1]] = 1
        interactions_to_preds = np.zeros((self.num_interactions, self.num_actions))
        interactions_to_preds[np.arange(self.num_interactions), interactions[:, 0]] = 1

        adj_an = torch.from_numpy(interactions_to_obj).float()
        adj_av = torch.from_numpy(interactions_to_preds).float()

        # Normalise.
        self.adj_an = nn.Parameter(torch.diag(1 / (1 + adj_an.sum(dim=1)).sqrt()) @ adj_an @ torch.diag(1 / (1 + adj_an.sum(dim=0)).sqrt()),
                                   requires_grad=False)
        self.adj_av = nn.Parameter(torch.diag(1 / (1 + adj_av.sum(dim=1)).sqrt()) @ adj_av @ torch.diag(1 / (1 + adj_av.sum(dim=0)).sqrt()),
                                   requires_grad=False)

        self.z_n = nn.Parameter(torch.empty(self.num_objects, input_repr_dim).normal_(), requires_grad=True)
        self.z_v = nn.Parameter(torch.empty(self.num_actions, input_repr_dim).normal_(), requires_grad=True)
        self.z_a = nn.Parameter(torch.empty(self.num_interactions, input_repr_dim).normal_(), requires_grad=True)

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

    def _forward(self):
        z_n = self.z_n
        z_v = self.z_v
        z_a = self.z_a
        for i in range(len(self.gc_layers)):
            prev_z_n, prev_z_v, prev_z_a = z_n, z_v, z_a
            z_n = self.gc_layers[i](prev_z_n + self.adj_an.t() @ prev_z_a)
            z_v = self.gc_layers[i](prev_z_v + self.adj_av.t() @ prev_z_a)
            z_a = self.gc_layers[i](prev_z_a + self.adj_an @ prev_z_n + self.adj_av @ prev_z_v)
        return z_a, z_v, z_n


class KatoGCNBranch(AbstractHOIBranch):
    def __init__(self, dataset: HicoSplit, gc_dims=(512, 200), **kwargs):
        super().__init__(**kwargs)
        self.word_emb_dim = 200

        full_dataset = dataset.full_dataset
        interactions = full_dataset.interactions  # each is [p, o]
        num_interactions = interactions.shape[0]
        assert num_interactions == 600
        interactions_to_obj = np.zeros((num_interactions, full_dataset.num_object_classes))
        interactions_to_obj[np.arange(num_interactions), interactions[:, 1]] = 1
        interactions_to_preds = np.zeros((num_interactions, full_dataset.num_actions))
        interactions_to_preds[np.arange(num_interactions), interactions[:, 0]] = 1

        adj_av = torch.from_numpy(interactions_to_preds).float()
        adj_an = torch.from_numpy(interactions_to_obj).float()
        adj_nn = torch.eye(full_dataset.num_object_classes).float()
        adj_vv = torch.eye(full_dataset.num_actions).float()

        # Normalise. The vv and nn matrices don't need it since they are identities. I think the other ones are supposed to be normalised like
        # this, but the paper is not clear at all.
        self.adj_vv = nn.Parameter(adj_vv, requires_grad=False)
        self.adj_nn = nn.Parameter(adj_nn, requires_grad=False)
        self.adj_an = nn.Parameter(torch.diag(1 / adj_an.sum(dim=1).sqrt()) @ adj_an @ torch.diag(1 / adj_an.sum(dim=0).sqrt()),
                                   requires_grad=False)
        self.adj_av = nn.Parameter(torch.diag(1 / adj_av.sum(dim=1).sqrt()) @ adj_av @ torch.diag(1 / adj_av.sum(dim=0).sqrt()),
                                   requires_grad=False)

        self.word_embs = WordEmbeddings(source='glove', dim=self.word_emb_dim, normalize=True)
        obj_word_embs = self.word_embs.get_embeddings(full_dataset.objects, retry='avg')
        pred_word_embs = self.word_embs.get_embeddings(full_dataset.predicates, retry='avg')

        self.z_n = nn.Parameter(torch.from_numpy(obj_word_embs).float(), requires_grad=False)
        self.z_v = nn.Parameter(torch.from_numpy(pred_word_embs).float(), requires_grad=False)

        self.gc_layers = nn.ModuleList([nn.Sequential(nn.Linear(self.word_emb_dim if i == 0 else gc_dims[i - 1], gc_dims[i]),
                                                      nn.ReLU(inplace=True),
                                                      nn.Dropout(p=0.5))
                                        for i in range(len(gc_dims))])

    def _forward(self):
        prev_z_n = self.z_n
        prev_z_v = self.z_v
        # z_a is 0

        # First layer is different for computational efficiency (z_a is 0)
        z_n = self.gc_layers[0](self.adj_nn @ prev_z_n)
        z_v = self.gc_layers[0](self.adj_vv @ prev_z_v)
        z_a = self.gc_layers[0](self.adj_an @ prev_z_n + self.adj_av @ prev_z_v)
        prev_z_n, prev_z_v, prev_z_a = z_n, z_v, z_a
        for i in range(1, len(self.gc_layers)):
            if i < len(self.gc_layers) - 1:
                z_n = self.gc_layers[i](self.adj_nn @ prev_z_n + self.adj_an.t() @ prev_z_a)
                z_v = self.gc_layers[i](self.adj_vv @ prev_z_v + self.adj_av.t() @ prev_z_a)
            z_a = self.gc_layers[i](prev_z_a + self.adj_an @ prev_z_n + self.adj_av @ prev_z_v)
            prev_z_n, prev_z_v, prev_z_a = z_n, z_v, z_a

        return z_a, z_v, z_n
