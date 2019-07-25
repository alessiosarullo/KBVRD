import os

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image

from config import cfg
from lib.dataset.hico.hico import Hico
from lib.dataset.utils import Splits
from lib.dataset.hoi_dataset import HoiDataset


class HicoSplit(HoiDataset):
    def __init__(self, split, hico: Hico, image_inds=None, object_inds=None, predicate_inds=None):
        self.hico = hico  # type: Hico
        self.split = split
        self.image_inds = image_inds

        object_inds = sorted(object_inds) if object_inds is not None else range(self.hico.num_object_classes)
        self.objects = [hico.objects[i] for i in object_inds]
        self.active_object_classes = np.array(object_inds, dtype=np.int)

        predicate_inds = sorted(predicate_inds) if predicate_inds is not None else range(self.hico.num_predicates)
        self.predicates = [hico.predicates[i] for i in predicate_inds]
        self.active_predicates = np.array(predicate_inds, dtype=np.int)

        self.active_interactions = np.array(sorted(set(np.unique(self.hico.op_pair_to_interaction[:, self.active_predicates]).tolist()) - {-1}),
                                            dtype=np.int)
        self.interactions = self.hico.interactions[self.active_interactions, :]  # original predicate and object inds

        self.img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        try:
            precomputed_feats_fn = cfg.precomputed_feats_format % ('hico', cfg.rcnn_arch, split.value)
            pc_feats_file = h5py.File(precomputed_feats_fn, 'r')
            self.pc_img_feats = pc_feats_file['img_feats'][:]
        except OSError:
            self.pc_img_feats = None

    @property
    def precomputed_visual_feat_dim(self):
        return self.pc_img_feats.shape[1]

    @property
    def human_class(self) -> int:
        return self.hico.human_class

    @property
    def num_object_classes(self):
        return len(self.objects)

    @property
    def num_predicates(self):
        return len(self.predicates)

    @property
    def num_interactions(self):
        return self.interactions.shape[0]

    @property
    def num_images(self):
        return self.hico.split_annotations[self.split].shape[0]

    def get_loader(self, batch_size, num_workers=0, num_gpus=1, shuffle=None, drop_last=True, **kwargs):
        def collate(idx_list):
            idxs = np.array(idx_list)
            feats = torch.tensor(self.pc_img_feats[idxs, :], device=device)
            if self.split != Splits.TEST:
                labels = self.hico.split_annotations[self.split][idxs, :]
                if self.active_interactions.size < self.hico.num_interactions:
                    all_labels = labels
                    labels = np.zeros_like(all_labels)
                    labels[self.active_interactions] = all_labels[self.active_interactions]
                labels = torch.tensor(labels, device=device)
            else:
                labels = None
            return feats, labels

        if self.pc_img_feats is None:
            raise NotImplementedError('This is only possible with precomputed features.')

        if shuffle is None:
            shuffle = True if self.split == Splits.TRAIN else False
        batch_size = batch_size * num_gpus

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        data_loader = torch.utils.data.DataLoader(
            dataset=self if self.image_inds is None else Subset(self, self.image_inds),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda x: collate(x),
            drop_last=drop_last,
            # pin_memory=True,  # disable this in case of freezes
            **kwargs,
        )
        return data_loader

    def get_img(self, img_id):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img = Image.open(os.path.join(self.hico.get_img_dir(self.split), self.hico.split_filenames[self.split][img_id])).convert('RGB')
        img = self.img_transform(img).to(device=device)
        return img

    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return self.num_images

    @classmethod
    def get_splits(cls, pred_inds=None, obj_inds=None):
        splits = {}
        hico = Hico()

        if obj_inds is not None:
            print(f'{Splits.TRAIN.value.capitalize()} objects:', obj_inds)
            assert hico.human_class in obj_inds
        if pred_inds is not None:
            print(f'{Splits.TRAIN.value.capitalize()} predicates:', pred_inds)
            assert 0 in pred_inds

        # Split train/val if needed
        if cfg.val_ratio > 0:
            num_imgs = len(hico.split_filenames[Splits.TRAIN])
            num_val_imgs = int(num_imgs * cfg.val_ratio)
            splits[Splits.TRAIN] = cls(split=Splits.TRAIN, hico=hico, image_inds=list(range(0, num_imgs - num_val_imgs)),
                                       object_inds=obj_inds, predicate_inds=pred_inds)
            splits[Splits.VAL] = cls(split=Splits.TRAIN, hico=hico, image_inds=list(range(num_imgs - num_val_imgs, num_imgs)),
                                     object_inds=obj_inds, predicate_inds=pred_inds)
        else:
            splits[Splits.TRAIN] = cls(split=Splits.TRAIN, hico=hico, object_inds=obj_inds, predicate_inds=pred_inds)
        splits[Splits.TEST] = cls(split=Splits.TEST, hico=hico)

        return splits
