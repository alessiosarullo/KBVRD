from typing import List

from config import cfg
from lib.dataset.hico.hico_split import HicoSplit
from lib.models.abstract_model import AbstractModel
from lib.models.branches import *
from lib.models.containers import Prediction
from lib.models.misc import bce_loss


class HicoBaseModel(AbstractModel):
    @classmethod
    def get_cline_name(cls):
        return 'hicobase'

    def __init__(self, dataset: HicoSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        assert cfg.hico
        self.dataset = dataset

        vis_feat_dim = self.dataset.precomputed_visual_feat_dim
        hidden_dim = 1024
        self.repr_dim = 1024

        self.hoi_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim, hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=cfg.dropout),
                                            nn.Linear(hidden_dim, self.repr_dim),
                                            ])
        nn.init.xavier_normal_(self.hoi_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.hoi_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('linear'))

        self.output_mlp = nn.Linear(self.repr_dim, dataset.hico.num_interactions, bias=False)
        torch.nn.init.xavier_normal_(self.output_mlp.weight, gain=1.0)

    def forward(self, x: List[torch.Tensor], inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):

            feats, labels = x
            output = self._forward(feats, labels)

            if not inference:
                zero_labels = (labels == 0)
                labels.clamp_(min=0)
                loss_mat = bce_loss(output, labels, reduce=False)
                if cfg.hico_lhard:
                    loss_mat[zero_labels] = 0
                losses = {'hoi_loss': loss_mat.sum(dim=1).mean()}
                return losses
            else:
                prediction = Prediction()
                prediction.hoi_scores = torch.sigmoid(output).cpu().numpy()
                return prediction

    def _forward(self, feats, labels):
        hoi_repr = self.hoi_repr_mlp(feats)
        output_logits = self.output_mlp(hoi_repr)
        return output_logits
