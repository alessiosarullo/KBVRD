import pickle

import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from config import cfg
from lib.dataset.hico import HicoSplit
from lib.dataset.utils import Splits
from lib.models.abstract_model import AbstractModel
from scripts.utils import get_all_models_by_name


def main():
    cfg.parse_args(fail_if_missing=False)
    cfg.load()

    if cfg.seenf >= 0:
        inds_dict = pickle.load(open(cfg.active_classes_file, 'rb'))
        act_inds = sorted(inds_dict[Splits.TRAIN.value]['act'].tolist())
        obj_inds = sorted(inds_dict[Splits.TRAIN.value]['obj'].tolist())
    else:
        obj_inds = act_inds = None

    splits = HicoSplit.get_splits(obj_inds=obj_inds, act_inds=act_inds)
    train_split, val_split, test_split = splits[Splits.TRAIN], splits[Splits.VAL], splits[Splits.TEST]

    # Model
    model = get_all_models_by_name()[cfg.model](train_split)  # type: AbstractModel

    ckpt = torch.load(cfg.saved_model_file, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])

    model.eval()
    _, act_class_embs = model.gcn()

    act_emb_2d = TSNE().fit_transform(act_class_embs)

    plt.scatter(act_emb_2d[:, 0], act_emb_2d[:, 1])
    plt.savefig('tsne.png', dpi=300)


if __name__ == '__main__':
    main()
