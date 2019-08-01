import os

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from config import cfg
from lib.dataset.hico.hico import Hico
from lib.dataset.hoi_dataset import HoiDatasetSplit
from lib.dataset.utils import Splits
from lib.stats.utils import Timer


class HicoSplit(HoiDatasetSplit):
    def __init__(self, split, hico: Hico, image_inds=None, object_inds=None, predicate_inds=None):
        self.full_dataset = hico  # type: Hico
        self.split = split
        self.image_inds = image_inds

        object_inds = sorted(object_inds) if object_inds is not None else range(self.full_dataset.num_object_classes)
        self.objects = [hico.objects[i] for i in object_inds]
        self.active_object_classes = np.array(object_inds, dtype=np.int)

        predicate_inds = sorted(predicate_inds) if predicate_inds is not None else range(self.full_dataset.num_actions)
        self.predicates = [hico.predicates[i] for i in predicate_inds]
        self.active_predicates = np.array(predicate_inds, dtype=np.int)

        active_op_mat = self.full_dataset.op_pair_to_interaction[self.active_object_classes, :][:, self.active_predicates]
        active_interactions = set(np.unique(active_op_mat).tolist()) - {-1}
        self.active_interactions = np.array(sorted(active_interactions), dtype=np.int)
        self.interactions = self.full_dataset.interactions[self.active_interactions, :]  # original predicate and object inds

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
        return self.full_dataset.human_class

    @property
    def num_object_classes(self):
        return len(self.objects)

    @property
    def num_actions(self):
        return len(self.predicates)

    @property
    def num_interactions(self):
        return self.interactions.shape[0]

    @property
    def num_images(self):
        return self.full_dataset.split_annotations[self.split].shape[0]

    def get_loader(self, batch_size, num_workers=0, num_gpus=1, shuffle=None, drop_last=True, **kwargs):
        def collate(idx_list):
            Timer.get('GetBatch').tic()
            idxs = np.array(idx_list)
            feats = torch.tensor(self.pc_img_feats[idxs, :], dtype=torch.float32, device=device)
            if self.split != Splits.TEST:
                labels = self.full_dataset.split_annotations[self.split][idxs, :]
                if self.active_interactions.size < self.full_dataset.num_interactions:
                    all_labels = labels
                    labels = np.zeros_like(all_labels)
                    labels[:, self.active_interactions] = all_labels[:, self.active_interactions]
                labels = torch.tensor(labels, dtype=torch.float32, device=device)
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
        img = Image.open(os.path.join(self.full_dataset.get_img_dir(self.split), self.full_dataset.split_filenames[self.split][img_id])).convert('RGB')
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

        tr = splits[Splits.TRAIN]
        if obj_inds is not None:
            print(f'{Splits.TRAIN.value.capitalize()} objects ({tr.active_object_classes.size}):', tr.active_object_classes.tolist())
            assert hico.human_class in obj_inds
        if pred_inds is not None:
            print(f'{Splits.TRAIN.value.capitalize()} predicates ({tr.active_predicates.size}):', tr.active_predicates.tolist())
            assert 0 in pred_inds
        if obj_inds is not None or pred_inds is not None:
            print(f'{Splits.TRAIN.value.capitalize()} interactions ({tr.active_interactions.size}):', tr.active_interactions.tolist())
            assert 0 in pred_inds

        return splits


class HicoHoiSplit(HicoSplit):
    def __init__(self, split, hico: Hico, image_inds=None, object_inds=None, predicate_inds=None):
        super().__init__(split, hico, image_inds, object_inds, predicate_inds)
        labels = self.full_dataset.split_annotations[self.split]
        pos_examples = np.stack(np.where(labels), axis=1)
        neg_examples = np.stack(np.where(labels <= 0), axis=1)  # no hard negative mining

        if image_inds is not None:
            image_mask = np.zeros(self.full_dataset.split_annotations[self.split].shape[0], dtype=np.bool)
            image_mask[image_inds] = 1
            pos_examples = pos_examples[image_mask[pos_examples[:, 0]], :]
            neg_examples = neg_examples[image_mask[neg_examples[:, 0]], :]
        if self.active_interactions.size < self.full_dataset.num_interactions:
            interaction_mask = np.zeros(self.full_dataset.split_annotations[self.split].shape[1], dtype=np.bool)
            interaction_mask[self.active_interactions] = 1
            pos_examples = pos_examples[interaction_mask[pos_examples[:, 1]], :]
            neg_examples = neg_examples[interaction_mask[neg_examples[:, 1]], :]

        self.num_pos = pos_examples.shape[0]
        self.num_neg = neg_examples.shape[0]
        self.examples = np.concatenate([pos_examples, neg_examples], axis=0)
        self.example_mask = np.zeros(self.examples.shape[0], dtype=bool)
        self.example_mask[:self.num_pos] = True

    def get_loader(self, batch_size, num_workers=0, num_gpus=1, shuffle=None, drop_last=True, **kwargs):
        def collate(idx_list):
            Timer.get('GetBatch').tic()
            idxs = np.array(idx_list)
            img_idxs = self.examples[idxs, 0]
            feats = torch.tensor(self.pc_img_feats[img_idxs, :], dtype=torch.float32, device=device)
            if self.split != Splits.TEST:
                labels = torch.tensor(self.examples[idxs, 1], device=device)
                label_mask = torch.tensor(self.example_mask[idxs].astype(np.uint8), dtype=torch.float32, device=device)
            else:
                labels = label_mask = None
            Timer.get('GetBatch').toc()
            return feats, labels, label_mask

        if self.pc_img_feats is None:
            raise NotImplementedError('This is only possible with precomputed features.')

        if shuffle is None:
            shuffle = True if self.split == Splits.TRAIN else False
        batch_size = batch_size * num_gpus

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        data_loader = torch.utils.data.DataLoader(
            dataset=self,
            batch_sampler=BalancedSampler(self, batch_size, drop_last, shuffle),
            num_workers=num_workers,
            collate_fn=lambda x: collate(x),
            # pin_memory=True,  # disable this in case of freezes
            **kwargs,
        )
        return data_loader

    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return self.examples.shape[0]

    @classmethod
    def get_splits(cls, pred_inds=None, obj_inds=None):
        splits = {}
        hico = Hico()

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
        splits[Splits.TEST] = HicoSplit(split=Splits.TEST, hico=hico)

        tr = splits[Splits.TRAIN]
        if obj_inds is not None:
            print(f'{Splits.TRAIN.value.capitalize()} objects ({tr.active_object_classes.size}):', tr.active_object_classes.tolist())
            assert hico.human_class in obj_inds
        if pred_inds is not None:
            print(f'{Splits.TRAIN.value.capitalize()} predicates ({tr.active_predicates.size}):', tr.active_predicates.tolist())
            assert 0 in pred_inds
        if obj_inds is not None or pred_inds is not None:
            print(f'{Splits.TRAIN.value.capitalize()} interactions ({tr.active_interactions.size}):', tr.active_interactions.tolist())
            assert 0 in pred_inds

        return splits


class BalancedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: HicoHoiSplit, hoi_batch_size, drop_last, shuffle):
        super().__init__(dataset)
        if not drop_last:
            raise NotImplementedError()

        self.batch_size = hoi_batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.dataset = dataset

        self.pos_samples = np.flatnonzero(dataset.example_mask)
        self.neg_samples = np.flatnonzero(dataset.example_mask == 0)

        self.neg_pos_ratio = cfg.hoi_bg_ratio
        pos_per_batch = hoi_batch_size / (self.neg_pos_ratio + 1)
        self.pos_per_batch = int(pos_per_batch)
        self.neg_per_batch = hoi_batch_size - self.pos_per_batch
        assert pos_per_batch == self.pos_per_batch
        assert self.neg_pos_ratio == int(self.neg_pos_ratio)

        self.batches = self.get_all_batches()

    def __iter__(self):
        for batch in self.batches:
            yield batch
        self.batches = self.get_all_batches()

    def __len__(self):
        return len(self.batches)

    def get_all_batches(self):
        batches = []

        # Positive samples
        pos_samples = np.random.permutation(self.pos_samples) if self.shuffle else self.pos_samples
        batch = []
        for sample in pos_samples:
            batch.append(sample)
            if len(batch) >= self.pos_per_batch:
                assert len(batch) == self.pos_per_batch
                batches.append(batch)
                batch = []

        # Negative samples
        neg_samples = []
        for n in range(int(np.ceil(self.neg_pos_ratio * self.pos_samples.shape[0] / self.neg_samples.shape[0]))):
            ns = np.random.permutation(self.neg_samples) if self.shuffle else self.neg_samples
            neg_samples.append(ns)
        neg_samples = np.concatenate(neg_samples, axis=0)
        batch_idx = 0
        for sample in neg_samples:
            if batch_idx == len(batches):
                break
            batch = batches[batch_idx]
            batch.append(sample)
            if len(batch) >= self.batch_size:
                assert len(batch) == self.batch_size
                batch_idx += 1
        assert batch_idx == len(batches)

        # Check
        for i, batch in enumerate(batches):
            assert len(batch) == self.batch_size, (i, len(batch), len(batches))

        return batches
