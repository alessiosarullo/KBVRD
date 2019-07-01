import numpy as np
import torch
import torch.utils.data
from typing import List
from collections import Counter

from config import cfg
from lib.dataset.hicodet.pc_hicodet_split import PrecomputedHicoDetSplit, PrecomputedMinibatch, PrecomputedExample
from lib.dataset.utils import Splits
from lib.stats.utils import Timer


class PrecomputedHicoDetHOISplit(PrecomputedHicoDetSplit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.split == Splits.TEST:
            raise ValueError('HOI-oriented dataset can only be used during training (labels are required to balance examples).')

        self.pc_im_idx_to_im_idx = {}
        for pc_im_idx, pc_im_id in enumerate(self.pc_image_ids):
            im_idx = np.flatnonzero(self.image_ids == pc_im_id).tolist()  # type: List
            assert len(im_idx) == 1, im_idx
            assert pc_im_id not in self.im_id_to_pc_im_idx
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

    def __getitem__(self, pc_hoi_idx) -> PrecomputedExample:
        Timer.get('GetBatch').tic()

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

        # HOI data
        entry.precomp_hoi_infos = self.pc_ho_infos[pc_hoi_idx, :].copy()
        entry.precomp_hoi_union_boxes = self.pc_union_boxes[pc_hoi_idx, :].copy()
        entry.precomp_action_labels = self.pc_action_labels[pc_hoi_idx, :].copy()

        # Object data
        img_box_inds = np.flatnonzero(self.pc_box_im_idxs == pc_im_idx)
        assert img_box_inds.size > 0
        box_start, box_end = img_box_inds[0], img_box_inds[-1] + 1
        assert np.all(img_box_inds == np.arange(box_start, box_end))
        entry.precomp_boxes_ext = self.pc_boxes_ext[box_start:box_end, :].copy()
        entry.precomp_box_feats = self.pc_boxes_feats[box_start:box_end, :].copy()
        entry.precomp_box_labels = self.pc_box_labels[box_start:box_end].copy()

        Timer.get('GetBatch').toc()
        return entry


class BalancedTripletSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: PrecomputedHicoDetHOISplit, hoi_batch_size, drop_last, shuffle):
        super().__init__(dataset)
        if not drop_last or not shuffle:
            raise NotImplementedError()

        self.batch_size = hoi_batch_size
        self.drop_last = drop_last
        self.dataset = dataset

        image_ids = set(dataset.image_ids)
        pos_hois_mask = np.any(dataset.pc_action_labels[:, 1:], axis=1)
        neg_hois_mask = (dataset.pc_action_labels[:, 0] > 0)
        if dataset.pc_hoi_mask is None:
            assert np.all(pos_hois_mask ^ neg_hois_mask)
        else:
            assert np.all(pos_hois_mask | neg_hois_mask | (~dataset.pc_hoi_mask))
            assert not np.any(pos_hois_mask & (~dataset.pc_hoi_mask))
            assert not np.any(neg_hois_mask & (~dataset.pc_hoi_mask))
            assert not np.any(pos_hois_mask & neg_hois_mask)
        split_mask = np.array([dataset.pc_image_ids[im_ind] in image_ids for im_ind in dataset.pc_ho_im_idxs])
        pos_hois_mask = pos_hois_mask & split_mask
        neg_hois_mask = neg_hois_mask & split_mask

        self.pos_sampler = torch.utils.data.SubsetRandomSampler(np.flatnonzero(pos_hois_mask))
        self.neg_sampler = torch.utils.data.SubsetRandomSampler(np.flatnonzero(neg_hois_mask))

        neg_pos_ratio = cfg.opt.hoi_bg_ratio
        pos_per_batch = hoi_batch_size / (neg_pos_ratio + 1)
        self.pos_per_batch = int(pos_per_batch)
        self.neg_per_batch = hoi_batch_size - self.pos_per_batch
        assert pos_per_batch == self.pos_per_batch
        assert neg_pos_ratio == int(neg_pos_ratio)

        self.batches = self.get_all_batches()

    def __iter__(self):
        for batch in self.batches:
            yield batch
        self.batches = self.get_all_batches()

    def __len__(self):
        return len(self.batches)

    def get_all_batches(self):
        batches = []
        batch = []
        for pc_hoi_idx in self.pos_sampler:
            batch.append(pc_hoi_idx)
            if len(batch) >= self.pos_per_batch:
                assert len(batch) == self.pos_per_batch
                batches.append(batch)
                batch = []

        batch_idx = 0
        for pc_hoi_idx in self.neg_sampler:
            batch = batches[batch_idx]
            batch.append(pc_hoi_idx)
            if len(batch) >= self.batch_size:
                batch_idx += 1

        # Check
        for batch in batches:
            assert len(batch) == self.batch_size

        return batches
