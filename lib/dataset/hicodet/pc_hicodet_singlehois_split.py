from builtins import super
from typing import List

import numpy as np
import torch
import torch.utils.data

from config import cfg
from lib.dataset.hicodet.pc_hicodet_split import PrecomputedHicoDetSplit, PrecomputedMinibatch, PrecomputedExample, PrecomputedFilesHandler
from lib.dataset.utils import Splits
from lib.stats.utils import Timer


class PrecomputedHicoDetSingleHOIsSplit(PrecomputedHicoDetSplit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.split == Splits.TEST:
            raise ValueError('HOI-oriented dataset can only be used during training (labels are required to balance examples).')

        if not cfg.program.save_mem:
            self.pc_boxes_feats = PrecomputedFilesHandler.get(self.precomputed_feats_fn, 'box_feats', load_in_memory=True)
            self.pc_union_boxes_feats =  PrecomputedFilesHandler.get(self.precomputed_feats_fn, 'union_boxes_feats', load_in_memory=True)

        self.pc_im_idx_to_im_idx = {}
        for pc_im_idx, pc_im_id in enumerate(self.pc_image_ids):
            im_idx = np.flatnonzero(self.image_ids == pc_im_id).tolist()  # type: List
            if im_idx:
                assert len(im_idx) == 1, im_idx
                self.pc_im_idx_to_im_idx[pc_im_idx] = im_idx[0]

        self.pc_im_box_range_inds = np.full((self.pc_image_ids.shape[0], 2), fill_value=-1, dtype=np.int)
        for i, pc_box_im_idx in enumerate(self.pc_box_im_idxs):
            if self.pc_im_box_range_inds[pc_box_im_idx, 0] < 0:
                self.pc_im_box_range_inds[pc_box_im_idx, 0] = i
            self.pc_im_box_range_inds[pc_box_im_idx, 1] = i

        # Check
        for i, (start, end) in enumerate(self.pc_im_box_range_inds):
            assert start >= 0 and end >= 0
            assert np.all(self.pc_box_im_idxs[start:end + 1] == i)

    def get_loader(self, batch_size, num_workers=0, num_gpus=1, shuffle=None, drop_last=True, **kwargs):
        if shuffle is None:
            shuffle = True if self.split == Splits.TRAIN else False
        batch_size = batch_size * num_gpus

        data_loader = torch.utils.data.DataLoader(
            dataset=self,
            batch_sampler=BalancedTripletMLSampler(self, batch_size, drop_last, shuffle),
            num_workers=num_workers,
            collate_fn=lambda x: self.collate(x),
            # pin_memory=True,  # disable this in case of freezes
            **kwargs,
        )
        return data_loader

    def __len__(self):
        return self.num_precomputed_hois

    def __getitem__(self, pc_hoi_idx: int) -> int:
        return pc_hoi_idx

    def collate(self, pc_hoi_idxs):
        minibatch = PrecomputedMinibatch()

        pc_hoi_idxs = np.array(pc_hoi_idxs)
        arange = np.arange(len(pc_hoi_idxs))

        pc_im_idxs = self.pc_ho_im_idxs[pc_hoi_idxs]
        img_infos = self.pc_image_infos[pc_im_idxs, :].copy()

        for i, pc_hoi_idx in enumerate(pc_hoi_idxs):
            Timer.get('GetBatch').tic()
            pc_im_idx = pc_im_idxs[i]
            im_idx = self.pc_im_idx_to_im_idx[pc_im_idx]
            img_id = self.image_ids[im_idx]
            assert self.pc_image_ids[pc_im_idx] == img_id, (self.pc_image_ids[pc_im_idx], img_id)
            minibatch.other_ex_data += [{'index': im_idx,
                                         'id': img_id,
                                         'fn': self._data[im_idx].filename,
                                         'split': self.split,
                                         'im_size': img_infos[i, :2],  # this is the original one and won't be changed
                                         'im_scale': img_infos[i, 2],
                                         }]
            Timer.get('GetBatch').toc()

        img_infos[:, 0] = max(img_infos[:, 0])
        img_infos[:, 1] = max(img_infos[:, 1])
        minibatch.img_infos = img_infos

        all_box_pair_inds = self.pc_im_box_range_inds[pc_im_idxs, :1] + self.pc_ho_infos[pc_hoi_idxs, 1:]
        assert all_box_pair_inds.shape[1] == 2 and np.all(all_box_pair_inds[:, 1] <= self.pc_im_box_range_inds[pc_im_idxs, 1])
        all_box_inds = all_box_pair_inds.flatten()
        assert all_box_inds.size == 2 * len(pc_hoi_idxs)

        boxes_ext = self.pc_boxes_ext[all_box_inds, :]
        boxes_ext[:, 0] = np.repeat(arange, 2)
        minibatch.pc_boxes_ext = boxes_ext
        minibatch.pc_box_feats = self.pc_boxes_feats[all_box_inds, :]
        minibatch.pc_box_labels = self.pc_box_labels[all_box_inds].copy()

        minibatch.pc_ho_infos = np.stack([arange, 2 * arange, 2 * arange + 1], axis=1)
        minibatch.pc_ho_union_boxes = self.pc_union_boxes[pc_hoi_idxs, :]
        minibatch.pc_ho_union_feats = self.pc_union_boxes_feats[pc_hoi_idxs, :]
        minibatch.pc_action_labels = self.pc_action_labels[pc_hoi_idxs, :].copy()

        assert minibatch.pc_boxes_ext.shape[0] == minibatch.pc_box_feats.shape[0] == minibatch.pc_box_labels.shape[0]
        assert minibatch.pc_ho_infos.shape[0] == minibatch.pc_ho_union_boxes.shape[0] == minibatch.pc_ho_union_feats.shape[0] == \
               minibatch.pc_action_labels.shape[0]

        return minibatch


class BalancedTripletMLSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: PrecomputedHicoDetSingleHOIsSplit, hoi_batch_size, drop_last, shuffle):
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
        self.pos_samples = np.flatnonzero(pos_hois_mask)

        neg_hois_mask = neg_hois_mask & split_mask
        self.neg_samples = np.flatnonzero(neg_hois_mask)

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
