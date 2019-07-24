import os

import cv2
import numpy as np
from torch.utils.data import Dataset, Subset
from typing import List

import h5py
import torch
from config import cfg
from lib.dataset.hico.hico import Hico
from lib.dataset.utils import Splits, preprocess_img


class ImgEntry:
    def __init__(self, img_id, filename, split):
        self.id = img_id
        self.filename = filename
        self.split = split

        self.interactions = None

        self.image = None
        self.img_size = None
        self.scale = None


class HicoSplit(Dataset):
    def __init__(self, split, hico: Hico, object_inds=None, predicate_inds=None):
        self.hico = hico  # type: Hico
        self.split = split

        object_inds = sorted(object_inds) if object_inds is not None else range(self.hico.num_object_classes)
        self.objects = [hico.objects[i] for i in object_inds]
        self.active_object_classes = np.array(object_inds, dtype=np.int)

        predicate_inds = sorted(predicate_inds) if object_inds is not None else range(self.hico.num_predicates)
        self.predicates = [hico.predicates[i] for i in predicate_inds]
        self.active_predicates = np.array(predicate_inds, dtype=np.int)

        self.active_interactions = np.array(sorted(set(np.unique(self.hico.op_pair_to_interaction[:, self.active_predicates]).tolist()) - {-1}),
                                            dtype=np.int)
        self.interactions = self.hico.interactions[self.active_interactions, :]  # original predicate and object inds

        try:
            precomputed_feats_fn = cfg.program.precomputed_feats_format % ('hico', cfg.model.rcnn_arch, split.value)
            self.pc_feats_file = h5py.File(precomputed_feats_fn, 'r')
            self.pc_img_feats = self.pc_feats_file['img_feats'][:]
            self.pc_img_infos = self.pc_feats_file['img_infos'][:]
            if 'labels' in self.pc_feats_file.keys():
                self.pc_labels = self.pc_feats_file['labels'][:]
            else:
                self.pc_labels = None
        except OSError:
            self.pc_feats_file = None

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
            if self.pc_labels is not None:
                labels = self.pc_labels[idxs, :]
                if self.active_interactions.size < self.hico.num_interactions:
                    all_labels = labels
                    labels = np.zeros_like(all_labels)
                    labels[self.active_interactions] = all_labels[self.active_interactions]
                labels = torch.tensor(labels, device=device)
            else:
                labels = None
            return feats, labels

        if self.pc_feats_file is None:
            raise NotImplementedError('This is only possible with precomputed features.')

        if shuffle is None:
            shuffle = True if self.split == Splits.TRAIN else False
        batch_size = batch_size * num_gpus
        if self.split == Splits.TEST and batch_size > 1:
            print('! Only single-image batches are supported during prediction. Batch size changed from %d to 1.' % batch_size)
            batch_size = 1

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        data_loader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda x: collate(x),
            drop_last=drop_last,
            # pin_memory=True,  # disable this in case of freezes
            **kwargs,
        )
        return data_loader

    def get_img_entry(self, img_id, read_img=True):
        img_fn = self.hico.split_filenames[self.split][img_id]
        entry = ImgEntry(img_id=img_id, filename=img_fn, split=self.split)
        entry.interactions = self.hico.split_annotations[self.split][img_id, :]
        if read_img:
            raw_image = cv2.imread(os.path.join(self.hico.get_img_dir(self.split), img_fn))
            img_h, img_w = raw_image.shape[:2]
            image, img_scale_factor = preprocess_img(raw_image)  # FIXME This resizes based on the SMALLEST side
            img_size = [img_h, img_w]

            entry.image = image
            entry.img_size = img_size
            entry.scale = img_scale_factor
        return entry

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

        train_split = cls(split=Splits.TRAIN, hico=hico, object_inds=obj_inds, predicate_inds=pred_inds)
        splits[Splits.TEST] = cls(split=Splits.TEST, hico=hico)

        # Split train/val if needed
        if cfg.data.val_ratio > 0:
            num_imgs = train_split.num_images
            num_val_imgs = int(num_imgs * cfg.data.val_ratio)
            splits[Splits.TRAIN] = Subset(train_split, range(0, num_imgs - num_val_imgs))
            splits[Splits.VAL] = Subset(train_split, range(num_imgs - num_val_imgs, num_imgs))
        else:
            splits[Splits.TRAIN] = train_split

        return splits
