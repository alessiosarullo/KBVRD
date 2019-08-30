import os

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from config import cfg
from lib.dataset.utils import Splits
from lib.stats.utils import Timer

from lib.dataset.hoi_dataset import HoiDataset


class AbstractHoiDatasetSplit(Dataset):
    @property
    def num_objects(self):
        raise NotImplementedError

    @property
    def num_actions(self):
        raise NotImplementedError

    @property
    def num_interactions(self):
        raise NotImplementedError

    @property
    def num_images(self):
        raise NotImplementedError

    def get_loader(self, batch_size, num_workers=0, num_gpus=1, shuffle=None, drop_last=True, **kwargs) -> torch.utils.data.DataLoader:
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class HoiDatasetSplit(AbstractHoiDatasetSplit):
    def __init__(self, split, full_dataset: HoiDataset, image_inds=None, object_inds=None, action_inds=None):
        self.full_dataset = full_dataset  # type: HoiDataset
        self.split = split
        self.image_inds = image_inds

        object_inds = sorted(object_inds) if object_inds is not None else range(self.full_dataset.num_objects)
        self.objects = [full_dataset.objects[i] for i in object_inds]
        self.active_object_classes = np.array(object_inds, dtype=np.int)

        action_inds = sorted(action_inds) if action_inds is not None else range(self.full_dataset.num_actions)
        self.actions = [full_dataset.actions[i] for i in action_inds]
        self.active_actions = np.array(action_inds, dtype=np.int)

        active_op_mat = self.full_dataset.op_pair_to_interaction[self.active_object_classes, :][:, self.active_actions]
        active_interactions = set(np.unique(active_op_mat).tolist()) - {-1}
        self.active_interactions = np.array(sorted(active_interactions), dtype=np.int)
        self.interactions = self.full_dataset.interactions[self.active_interactions, :]  # original action and object inds

        self.img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        try:
            pc_feats_file = h5py.File(self._get__precomputed_feats_fn(split), 'r')
            self.pc_img_feats = pc_feats_file['img_feats'][:]
        except OSError:
            self.pc_img_feats = None

        self.labels = self.full_dataset.split_annotations[self.split]
        if self.active_interactions.size < self.full_dataset.num_interactions:
            all_labels = self.labels
            self.labels = np.zeros_like(all_labels)
            self.labels[:, self.active_interactions] = all_labels[:, self.active_interactions]

        if self.image_inds is None:
            self.non_empty_imgs = np.flatnonzero(np.any(self.labels, axis=1))
        else:
            image_mask = np.zeros(self.full_dataset.split_annotations[self.split].shape[0], dtype=np.bool)
            image_mask[self.image_inds] = True
            self.non_empty_imgs = np.flatnonzero(np.any(self.labels, axis=1) & image_mask)

    @property
    def precomputed_visual_feat_dim(self):
        return self.pc_img_feats.shape[1]

    @property
    def num_objects(self):
        return len(self.objects)

    @property
    def num_actions(self):
        return len(self.actions)

    @property
    def num_interactions(self):
        return self.interactions.shape[0]

    @property
    def num_images(self):
        return self.full_dataset.split_annotations[self.split].shape[0]

    def get_loader(self, batch_size, num_workers=0, num_gpus=1, shuffle=None, drop_last=True, **kwargs):
        # noinspection PyCallingNonCallable,PyUnresolvedReferences
        def collate(idx_list):
            Timer.get('GetBatch').tic()
            idxs = np.array(idx_list)
            feats = torch.tensor(self.pc_img_feats[idxs, :], dtype=torch.float32, device=device)
            if self.split != Splits.TEST:
                labels = torch.tensor(self.labels[idxs, :], dtype=torch.float32, device=device)
            else:
                labels = None
            Timer.get('GetBatch').toc()
            return feats, labels

        if self.pc_img_feats is None:
            raise NotImplementedError('This is only possible with precomputed features.')

        if shuffle is None:
            shuffle = True if self.split == Splits.TRAIN else False
        batch_size = batch_size * num_gpus

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.split == Splits.TEST:
            ds = self
        else:
            if cfg.filter_bg_only:
                ds = self if self.non_empty_imgs is None else Subset(self, self.non_empty_imgs)
            else:
                ds = self if self.image_inds is None else Subset(self, self.image_inds)
        data_loader = torch.utils.data.DataLoader(
            dataset=ds,
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
        img_fn = os.path.join(self.full_dataset.get_img_dir(self.split), self.full_dataset.split_filenames[self.split][img_id])
        img = Image.open(img_fn).convert('RGB')
        img = self.img_transform(img).to(device=device)
        return img

    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return self.num_images

    def _get__precomputed_feats_fn(self, split):
        raise NotImplementedError

    @classmethod
    def get_full_dataset(cls) -> HoiDataset:
        raise NotImplementedError

    @classmethod
    def get_splits(cls, act_inds=None, obj_inds=None):
        splits = {}
        full_dataset = cls.get_full_dataset()

        # Split train/val if needed
        if cfg.val_ratio > 0:
            num_imgs = len(full_dataset.split_filenames[Splits.TRAIN])
            num_val_imgs = int(num_imgs * cfg.val_ratio)
            splits[Splits.TRAIN] = cls(split=Splits.TRAIN, full_dataset=full_dataset, image_inds=list(range(0, num_imgs - num_val_imgs)),
                                       object_inds=obj_inds, action_inds=act_inds)
            splits[Splits.VAL] = cls(split=Splits.TRAIN, full_dataset=full_dataset, image_inds=list(range(num_imgs - num_val_imgs, num_imgs)),
                                     object_inds=obj_inds, action_inds=act_inds)
        else:
            splits[Splits.TRAIN] = cls(split=Splits.TRAIN, full_dataset=full_dataset, object_inds=obj_inds, action_inds=act_inds)
        splits[Splits.TEST] = cls(split=Splits.TEST, full_dataset=full_dataset)

        tr = splits[Splits.TRAIN]
        if obj_inds is not None:
            print(f'{Splits.TRAIN.value.capitalize()} objects ({tr.active_object_classes.size}):', tr.active_object_classes.tolist())
        if act_inds is not None:
            print(f'{Splits.TRAIN.value.capitalize()} actions ({tr.active_actions.size}):', tr.active_actions.tolist())
        if obj_inds is not None or act_inds is not None:
            print(f'{Splits.TRAIN.value.capitalize()} interactions ({tr.active_interactions.size}):', tr.active_interactions.tolist())

        return splits
