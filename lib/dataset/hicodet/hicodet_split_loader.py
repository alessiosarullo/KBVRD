from typing import List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from config import cfg
from lib.dataset.hicodet.hicodet_split import HicoDetSplit
from lib.dataset.utils import Splits, PrecomputedExample, PrecomputedMinibatch
from lib.stats.utils import Timer


class PrecomputedHicoDetSplit(HicoDetSplit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        precomputed_feats_fn = cfg.program.precomputed_feats_file_format % (cfg.model.rcnn_arch,
                                                                            (Splits.TEST if self.split == Splits.TEST else Splits.TRAIN).value)
        print('Loading precomputed feats for %s split from %s.' % (self.split.value, precomputed_feats_fn))
        self.pc_feats_file = h5py.File(precomputed_feats_fn, 'r')
        assert self.pc_feats_file['box_feats'].shape[1] == self.pc_feats_file['union_boxes_feats'].shape[1] == 2048  # FIXME magic constant

        self.pc_image_ids = self.pc_feats_file['image_ids'][:]
        self.pc_image_infos = self.pc_feats_file['img_infos'][:]

        self.pc_boxes_ext = self.pc_feats_file['boxes_ext'][:]
        self.pc_box_im_inds = self.pc_boxes_ext[:, 0].astype(np.int)
        try:
            self.pc_box_labels = self.pc_feats_file['box_labels'][:]
        except KeyError:
            self.pc_box_labels = None

        self.pc_ho_infos = self.pc_feats_file['ho_infos'][:].astype(np.int)
        self.pc_union_boxes = self.pc_feats_file['union_boxes'][:]
        self.pc_ho_im_inds = self.pc_ho_infos[:, 0]
        try:
            self.pc_action_labels = self.pc_feats_file['action_labels'][:]
        except KeyError:
            self.pc_action_labels = None

        # Map image IDs to indices over the precomputed image IDs
        assert len(set(self.image_ids) - set(self.pc_image_ids.tolist())) == 0
        assert len(self.pc_image_ids) == len(set(self.pc_image_ids))
        self.im_id_to_pc_im_idx = {}
        for im_id in self.image_ids:
            pc_im_idx = np.flatnonzero(self.pc_image_ids == im_id).tolist()  # type: List
            assert len(pc_im_idx) == 1, pc_im_idx
            assert im_id not in self.im_id_to_pc_im_idx
            self.im_id_to_pc_im_idx[im_id] = pc_im_idx[0]

    @property
    def precomputed_visual_feat_dim(self):
        return self.pc_feats_file['box_feats'].shape[1]

    @property
    def num_precomputed_hois(self):
        return self.pc_ho_infos.shape[0]

    def get_loader(self, batch_size, num_workers=0, num_gpus=1, shuffle=None, drop_last=True, **kwargs):
        if shuffle is None:
            shuffle = True if self.split == Splits.TRAIN else False
        batch_size = batch_size * num_gpus
        if self.split == Splits.TEST and batch_size > 1:
            print('! Only single-image batches are supported during prediction. Batch size changed from %d to 1.' % batch_size)
            batch_size = 1

        data_loader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda x: PrecomputedMinibatch.collate(x),
            drop_last=drop_last,
            # pin_memory=True,  # disable this in case of freezes
            **kwargs,
        )
        return data_loader

    def __getitem__(self, idx) -> PrecomputedExample:
        Timer.get('GetBatch').tic()
        im_data = self._data[idx]
        img_id = self.image_ids[idx]
        pc_im_idx = self.im_id_to_pc_im_idx[img_id]
        assert self.pc_image_ids[pc_im_idx] == img_id, (self.pc_image_ids[pc_im_idx], img_id)

        entry = PrecomputedExample(idx_in_split=idx, img_id=self.image_ids[idx], filename=im_data.filename, split=self.split)

        # Image data
        img_infos = self.pc_image_infos[pc_im_idx].copy()
        assert img_infos.shape == (3,)
        entry.img_size = img_infos[:2]
        entry.scale = img_infos[2]

        # Object data
        img_box_inds = np.flatnonzero(self.pc_box_im_inds == pc_im_idx)
        if img_box_inds.size > 0:
            start, end = img_box_inds[0], img_box_inds[-1] + 1
            assert np.all(img_box_inds == np.arange(start, end))  # slicing is much more efficient with H5 files

            precomp_boxes_ext = self.pc_boxes_ext[start:end, :]
            precomp_box_feats = self.pc_feats_file['box_feats'][start:end, :]
            precomp_masks = self.pc_feats_file['masks'][start:end, :, :]
            box_inds = None
            if self.pc_box_labels is not None:
                # TODO mapping of obj inds for rels
                precomp_box_labels = self.pc_box_labels[start:end].copy()

                if self.obj_class_inds.size < self.num_object_classes:
                    precomp_boxes_ext = precomp_boxes_ext[:, np.concatenate([np.arange(5), 5 + self.obj_class_inds])]
                    num_boxes = precomp_box_labels.shape[0]
                    bg_box_mask = (precomp_box_labels < 0)

                    precomp_box_labels_one_hot = np.zeros([num_boxes, len(self._hicodet.objects)], dtype=precomp_box_labels.dtype)
                    precomp_box_labels_one_hot[np.arange(num_boxes), precomp_box_labels] = 1
                    precomp_box_labels_one_hot[bg_box_mask, :] = 0
                    precomp_box_labels_one_hot = precomp_box_labels_one_hot[:, self.obj_class_inds]
                    feasible_box_labels_inds = np.any(precomp_box_labels_one_hot, axis=1) | bg_box_mask

                    box_inds = np.full_like(precomp_box_labels, fill_value=-1)
                    box_inds[feasible_box_labels_inds] = np.arange(np.sum(feasible_box_labels_inds))
                    precomp_boxes_ext = precomp_boxes_ext[feasible_box_labels_inds]
                    precomp_box_feats = precomp_box_feats[feasible_box_labels_inds]
                    precomp_masks = precomp_masks[feasible_box_labels_inds]
                    precomp_box_labels_one_hot = precomp_box_labels_one_hot[feasible_box_labels_inds]
                    precomp_box_labels = np.argmax(precomp_box_labels_one_hot, axis=1).astype(precomp_box_labels_one_hot.dtype)
                    precomp_box_labels[~np.any(precomp_box_labels_one_hot, axis=1)] = -1
            else:
                precomp_box_labels = None

            # HOI data
            img_hoi_inds = np.flatnonzero(self.pc_ho_im_inds == pc_im_idx)
            if img_hoi_inds.size > 0:
                start, end = img_hoi_inds[0], img_hoi_inds[-1] + 1
                assert np.all(img_hoi_inds == np.arange(start, end))  # slicing is much more efficient with H5 files
                precomp_hoi_infos = self.pc_ho_infos[start:end, :].copy()
                precomp_hoi_union_boxes = self.pc_union_boxes[start:end, :]
                precomp_hoi_union_feats = self.pc_feats_file['union_boxes_feats'][start:end, :]
                if self.pc_action_labels is not None:
                    precomp_action_labels = self.pc_action_labels[start:end, :]
                else:
                    precomp_action_labels = None

                if precomp_action_labels is not None:
                    assert precomp_box_labels is not None

                    if self.action_class_inds.size < self.num_predicates:
                        precomp_action_labels = precomp_action_labels[:, self.action_class_inds]

                    # Remap HOIs box indices
                    if box_inds is not None:
                        precomp_hoi_infos[:, 1] = box_inds[precomp_hoi_infos[:, 1]]
                        precomp_hoi_infos[:, 2] = box_inds[precomp_hoi_infos[:, 2]]

                    # Filter out HOIs
                    if self.action_class_inds.size < self.num_predicates or box_inds is not None:
                        feasible_hoi_labels_inds = np.any(precomp_action_labels, axis=1) & np.all(precomp_hoi_infos >= 0, axis=1)
                        assert np.any(feasible_hoi_labels_inds), (idx, pc_im_idx, img_id, im_data.filename)
                        precomp_hoi_infos = precomp_hoi_infos[feasible_hoi_labels_inds]
                        precomp_hoi_union_boxes = precomp_hoi_union_boxes[feasible_hoi_labels_inds]
                        precomp_hoi_union_feats = precomp_hoi_union_feats[feasible_hoi_labels_inds]
                        precomp_action_labels = precomp_action_labels[feasible_hoi_labels_inds, :]
                        assert np.all(np.sum(precomp_action_labels, axis=1) >= 1), precomp_action_labels

                        # Filter out boxes without interactions
                        hoi_box_inds = np.unique(precomp_hoi_infos[:, 1:])
                        precomp_boxes_ext = precomp_boxes_ext[hoi_box_inds]
                        precomp_box_feats = precomp_box_feats[hoi_box_inds]
                        precomp_masks = precomp_masks[hoi_box_inds]
                        precomp_box_labels = precomp_box_labels[hoi_box_inds]
                        box_inds = np.full(np.amax(hoi_box_inds) + 1, fill_value=-1)
                        box_inds[hoi_box_inds] = np.arange(hoi_box_inds.shape[0])
                        precomp_hoi_infos[:, 1] = box_inds[precomp_hoi_infos[:, 1]]
                        precomp_hoi_infos[:, 2] = box_inds[precomp_hoi_infos[:, 2]]
                        assert np.all(precomp_hoi_infos >= 0), precomp_hoi_infos

                assert np.all(precomp_hoi_infos >= 0)
                entry.precomp_action_labels = precomp_action_labels
                entry.precomp_hoi_infos = precomp_hoi_infos
                entry.precomp_hoi_union_boxes = precomp_hoi_union_boxes
                entry.precomp_hoi_union_feats = precomp_hoi_union_feats
            else:
                assert self.split == Splits.TEST, (idx, pc_im_idx, img_id, im_data.filename)

            entry.precomp_boxes_ext = precomp_boxes_ext
            entry.precomp_box_feats = precomp_box_feats
            entry.precomp_masks = precomp_masks
            entry.precomp_box_labels = precomp_box_labels
        assert (entry.precomp_box_labels is None and entry.precomp_action_labels is None) or \
               (entry.precomp_box_labels is not None and entry.precomp_action_labels is not None)
        Timer.get('GetBatch').toc()
        return entry

    def __len__(self):
        return self.num_images


class PrecomputedHOIHicoDetSplit(PrecomputedHicoDetSplit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.split == Splits.TEST:
            raise ValueError('HOI-oriented dataset can only be used during training (labels are required to balance examples).')
        self.pc_im_idx_to_im_idx = {}
        im_id_to_im_idx = {im_id: i for i, im_id in enumerate(self.image_ids)}
        for pc_im_idx, im_id in enumerate(self.pc_image_ids):
            if im_id in im_id_to_im_idx.keys():
                im_idx = im_id_to_im_idx[im_id]
                assert im_idx not in self.pc_im_idx_to_im_idx.values()
                self.pc_im_idx_to_im_idx[pc_im_idx] = im_idx

    def get_loader(self, batch_size, num_workers=0, num_gpus=1, shuffle=None, drop_last=True, **kwargs):
        if shuffle is None:
            shuffle = True if self.split == Splits.TRAIN else False
        batch_size = batch_size * num_gpus

        data_loader = torch.utils.data.DataLoader(
            dataset=self,
            batch_sampler=BalancedHOISampler(self, batch_size, drop_last, shuffle),
            num_workers=num_workers,
            collate_fn=lambda x: PrecomputedMinibatch.collate(x),
            # pin_memory=True,  # disable this in case of freezes
            **kwargs,
        )
        return data_loader

    def __getitem__(self, hoi_idx) -> PrecomputedExample:
        Timer.get('GetBatch').tic()
        pc_im_idx = self.pc_ho_im_inds[hoi_idx]
        img_id = self.pc_image_ids[pc_im_idx]
        im_idx = self.pc_im_idx_to_im_idx[pc_im_idx]
        im_data = self._data[im_idx]

        entry = PrecomputedExample(idx_in_split=im_idx, img_id=img_id, filename=im_data.filename, split=self.split)

        # Image data
        img_infos = self.pc_image_infos[pc_im_idx].copy()
        assert img_infos.shape == (3,)
        entry.img_size = img_infos[:2]
        entry.scale = img_infos[2]

        # HOI data
        precomp_hoi_infos = self.pc_ho_infos[hoi_idx, :].copy()
        precomp_hoi_union_boxes = self.pc_union_boxes[hoi_idx, :]
        precomp_hoi_union_feats = self.pc_feats_file['union_boxes_feats'][hoi_idx, :]
        if self.pc_action_labels is not None:
            precomp_action_labels = self.pc_action_labels[hoi_idx, :]
        else:
            precomp_action_labels = None

        # Object data
        img_box_inds = np.flatnonzero(self.pc_box_im_inds == pc_im_idx)
        assert img_box_inds.size > 0
        start, end = img_box_inds[0], img_box_inds[-1] + 1
        assert np.all(img_box_inds == np.arange(start, end))  # slicing is much more efficient with H5 files
        assert precomp_hoi_infos[1] < end and precomp_hoi_infos[2] < end

        pair = start + precomp_hoi_infos[1:]
        if pair[0] < pair[1]:
            precomp_hoi_infos[1] = 0
            precomp_hoi_infos[2] = 1
        else:
            pair = pair[[1, 0]]
            precomp_hoi_infos[1] = 1
            precomp_hoi_infos[2] = 0
        precomp_boxes_ext = self.pc_boxes_ext[pair, :]
        precomp_box_feats = self.pc_feats_file['box_feats'][pair, :]
        precomp_masks = self.pc_feats_file['masks'][pair, :, :]
        if self.pc_box_labels is not None:
            precomp_box_labels = self.pc_box_labels[pair].copy()
        else:
            precomp_box_labels = None

        # Create entry
        entry.precomp_action_labels = precomp_action_labels
        entry.precomp_hoi_infos = precomp_hoi_infos
        entry.precomp_hoi_union_boxes = precomp_hoi_union_boxes
        entry.precomp_hoi_union_feats = precomp_hoi_union_feats

        entry.precomp_boxes_ext = precomp_boxes_ext
        entry.precomp_box_feats = precomp_box_feats
        entry.precomp_masks = precomp_masks
        entry.precomp_box_labels = precomp_box_labels
        assert (entry.precomp_box_labels is None and entry.precomp_action_labels is None) or \
               (entry.precomp_box_labels is not None and entry.precomp_action_labels is not None)
        Timer.get('GetBatch').toc()
        return entry

    def __len__(self):
        return self.num_precomputed_hois


class BalancedHOISampler(torch.utils.data.Sampler):
    def __init__(self, dataset: PrecomputedHOIHicoDetSplit, hoi_batch_size, drop_last, shuffle):
        super().__init__(dataset)
        if not drop_last:
            raise NotImplementedError()
        self.hoi_batch_size = hoi_batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        image_ids = set(dataset.image_ids)
        split_mask = np.array([dataset.pc_image_ids[im_ind] in image_ids for im_ind in dataset.pc_ho_im_inds])
        pos_hois_mask = np.any(dataset.pc_action_labels[:, 1:], axis=1)
        neg_hois_mask = (dataset.pc_action_labels[:, 0] > 0)
        assert np.all(pos_hois_mask != neg_hois_mask)
        self.pos_hois = np.flatnonzero(pos_hois_mask & split_mask)
        self.neg_hois = np.flatnonzero(neg_hois_mask & split_mask)

        self.pos_per_batch = np.ceil(hoi_batch_size * self.pos_hois.size / self.neg_hois.size).astype(np.int)
        self.neg_per_batch = hoi_batch_size - self.pos_per_batch

        self.batches = self.get_all_batches()

    def __iter__(self):
        for batch in self.batches:
            yield batch
        self.batches = self.get_all_batches()

    def get_all_batches(self):
        pos_sampler = torch.utils.data.SubsetRandomSampler(self.pos_hois) if self.shuffle else torch.utils.data.SequentialSampler(self.pos_hois)
        neg_sampler = torch.utils.data.SubsetRandomSampler(self.neg_hois) if self.shuffle else torch.utils.data.SequentialSampler(self.neg_hois)
        neg_sampler_iter = iter(neg_sampler)
        batches = []
        batch = []
        for idx in pos_sampler:
            batch.append(idx)
            if len(batch) >= self.pos_per_batch:
                assert len(batch) == self.pos_per_batch
                for i in range(self.neg_per_batch):
                    batch.append(next(neg_sampler_iter))
                assert len(batch) == self.hoi_batch_size
                batches.append(batch)
                batch = []
        return batches

    def __len__(self):
        return len(self.batches)


def main():
    import random
    seed = 3 if not cfg.program.randomize else np.random.randint(1_000_000_000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    s = PrecomputedHOIHicoDetSplit.get_split(split=Splits.TRAIN)
    ld = s.get_loader(batch_size=64)

    for x in ld:
        pass


if __name__ == '__main__':
    main()