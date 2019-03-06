from typing import Dict

import torch
from torch import nn as nn


class AbstractHOIModel(nn.Module):
    @classmethod
    def get_cline_name(cls) -> str:
        raise NotImplementedError()

    def __init__(self, dataset, **kwargs):
        super().__init__()
        self.__dict__.update({k: v for k, v in kwargs.items() if k in self.__dict__.keys() and v is not None})
        self.dataset = dataset

    def get_losses(self, batch, **kwargs):
        raise NotImplementedError()

    def forward(self, x, predict=True, **kwargs):
        raise NotImplementedError()


class AbstractHOIBranch(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__dict__.update({k: v for k, v in kwargs.items() if k in self.__dict__.keys() and v is not None})
        self.values_to_monitor = {}  # type: Dict[str, torch.Tensor]

    def forward(self, *args, **kwargs):
        with torch.set_grad_enabled(self.training):
            return self._forward(*args, **kwargs)

    def _forward(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def output_dim(self):
        raise NotImplementedError()
