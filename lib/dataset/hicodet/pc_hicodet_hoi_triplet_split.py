import numpy as np
import torch
import torch.utils.data
from typing import List
from collections import Counter

from config import cfg
from lib.dataset.hicodet.pc_hicodet_split import PrecomputedHicoDetSplit, PrecomputedMinibatch, PrecomputedExample
from lib.dataset.utils import Splits
from lib.stats.utils import Timer


class PrecomputedHicoDetPureHOISplit(PrecomputedHicoDetSplit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.split == Splits.TEST:
            raise ValueError('HOI-oriented dataset can only be used during training (labels are required to balance examples).')

        self.pc_im_idx_to_im_idx = {}
        for pc_im_idx, pc_im_id in enumerate(self.pc_image_ids):
            im_idx = np.flatnonzero(self.image_ids == pc_im_id).tolist()  # type: List
            if im_idx:
                assert len(im_idx) == 1, im_idx
                assert pc_im_id not in self.pc_im_idx_to_im_idx
                self.pc_im_idx_to_im_idx[pc_im_idx] = im_idx[0]

    def get_loader(self, batch_size, num_workers=0, num_gpus=1, shuffle=None, drop_last=True, **kwargs):
        if shuffle is None:
            shuffle = True if self.split == Splits.TRAIN else False
        batch_size = batch_size * num_gpus

        data_loader = torch.utils.data.DataLoader(
            dataset=self,
            batch_sampler=BalancedTripletSampler(self, batch_size, drop_last, shuffle),
            num_workers=num_workers,
            collate_fn=lambda x: PrecomputedMinibatch.collate(x),
            # pin_memory=True,  # disable this in case of freezes
            **kwargs,
        )
        return data_loader

    def __len__(self):
        return self.num_precomputed_hois

    def __getitem__(self, idx) -> PrecomputedExample:
        Timer.get('GetBatch').tic()
        pc_hoi_idx, hoi_label = idx

        pc_im_idx = self.pc_ho_im_idxs[pc_hoi_idx]
        im_idx = self.pc_im_idx_to_im_idx[pc_im_idx]
        im_data = self._data[im_idx]
        img_id = self.image_ids[im_idx]
        assert self.pc_image_ids[pc_im_idx] == img_id, (self.pc_image_ids[pc_im_idx], img_id)

        # Image data
        entry = PrecomputedExample(idx_in_split=im_idx, img_id=self.image_ids[im_idx], filename=im_data.filename, split=self.split)
        img_infos = self.pc_image_infos[pc_im_idx].copy()
        assert img_infos.shape == (3,)
        entry.img_size = img_infos[:2]
        entry.scale = img_infos[2]

        hoi_infos = self.pc_ho_infos[pc_hoi_idx, :].copy()

        # Object data
        img_box_inds = np.flatnonzero(self.pc_box_im_idxs == pc_im_idx)
        assert img_box_inds.size > 0
        box_start, box_end = img_box_inds[0], img_box_inds[-1] + 1
        assert np.all(img_box_inds == np.arange(box_start, box_end))
        box_pair_inds = box_start + hoi_infos[1:]
        assert box_pair_inds.size == 2 and np.all(box_pair_inds < box_end)
        entry.precomp_boxes_ext = self.pc_boxes_ext[box_pair_inds, :].copy()
        entry.precomp_box_feats = self.pc_boxes_feats[box_pair_inds, :].copy()
        entry.precomp_box_labels = self.pc_box_labels[box_pair_inds].copy()

        # HOI data
        hoi_infos[1:] = [0, 1]
        entry.precomp_hoi_infos = hoi_infos[None, :]
        entry.precomp_hoi_union_boxes = self.pc_union_boxes[[pc_hoi_idx], :].copy()
        precomp_action_labels = self.pc_action_labels[pc_hoi_idx, :].copy()
        assert precomp_action_labels[hoi_label] == 1, (pc_hoi_idx, hoi_label, precomp_action_labels)
        precomp_action_labels[:] = 0
        precomp_action_labels[hoi_label] = 1
        entry.precomp_action_labels = precomp_action_labels[None, :]

        Timer.get('GetBatch').toc()
        return entry


class BalancedTripletSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: PrecomputedHicoDetPureHOISplit, hoi_batch_size, drop_last, shuffle):
        super().__init__(dataset)
        if not drop_last:
            raise NotImplementedError()

        self.batch_size = hoi_batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.dataset = dataset

        pos_hois_mask = np.any(dataset.pc_action_labels[:, 1:], axis=1)
        neg_hois_mask = (dataset.pc_action_labels[:, 0] > 0)
        if dataset.pc_hoi_mask is None:
            assert np.all(pos_hois_mask ^ neg_hois_mask)
        else:
            assert np.all(pos_hois_mask | neg_hois_mask | (~dataset.pc_hoi_mask))
            assert not np.any(pos_hois_mask & (~dataset.pc_hoi_mask))
            assert not np.any(neg_hois_mask & (~dataset.pc_hoi_mask))
            assert not np.any(pos_hois_mask & neg_hois_mask)

        pc_ho_im_ids = dataset.pc_image_ids[dataset.pc_ho_im_idxs]
        split_ids_mask = np.zeros(max(np.max(pc_ho_im_ids), np.max(dataset.image_ids)) + 1, dtype=bool)
        split_ids_mask[dataset.image_ids] = True
        split_mask = split_ids_mask[pc_ho_im_ids]

        pos_hois_mask = pos_hois_mask & split_mask
        pos_samples_mask_inds, pos_samples_labels = np.where(self.dataset.pc_action_labels[pos_hois_mask, :])
        pos_hois_ids = np.flatnonzero(pos_hois_mask)
        self.pos_samples = np.stack([pos_hois_ids[pos_samples_mask_inds], pos_samples_labels], axis=1)
        assert np.all(self.pos_samples[:, 1] > 0)

        neg_hois_mask = neg_hois_mask & split_mask
        neg_samples_mask_inds, neg_samples_labels = np.where(self.dataset.pc_action_labels[neg_hois_mask, :])
        neg_hois_ids = np.flatnonzero(neg_hois_mask)
        self.neg_samples = np.stack([neg_hois_ids[neg_samples_mask_inds], neg_samples_labels], axis=1)
        assert np.all(self.neg_samples[:, 1] == 0)

        self.neg_pos_ratio = cfg.opt.hoi_bg_ratio
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
            batch.append(sample.tolist())
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
            batch.append(sample.tolist())
            if len(batch) >= self.batch_size:
                assert len(batch) == self.batch_size
                batch_idx += 1
        assert batch_idx == len(batches)

        # Check
        for i, batch in enumerate(batches):
            assert len(batch) == self.batch_size, (i, len(batch), len(batches))

        return batches
