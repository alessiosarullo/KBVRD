from typing import List


import numpy as np
import torch
import torch.utils.data

from config import cfg
from lib.dataset.hicodet.hicodet_split import PrecomputedFilesHandler, HicoDetSplit, HicoDet, PrecomputedMinibatch
from lib.dataset.utils import Splits
from lib.utils import Timer


class HicoDetSingleHOIsSplit(HicoDetSplit):
    def __init__(self, split, full_dataset: HicoDet, image_inds=None, object_inds=None, action_inds=None):
        if split == Splits.TEST:
            raise ValueError('HOI-oriented dataset can only be used during training (labels are required to balance examples).')
        super().__init__(split, full_dataset, image_inds=image_inds)

        #############################################################################################################################
        # Filter out zero-shot indices
        #############################################################################################################################

        object_inds = sorted(object_inds or range(self.full_dataset.num_objects))
        self.objects = [full_dataset.objects[i] for i in object_inds]
        self.active_objects = np.array(object_inds, dtype=np.int)

        action_inds = sorted(action_inds or range(self.full_dataset.num_actions))
        self.actions = [full_dataset.actions[i] for i in action_inds]
        self.active_actions = np.array(action_inds, dtype=np.int)

        active_interactions = set(np.unique(self.full_dataset.oa_pair_to_interaction[:, self.active_actions]).tolist()) - {-1}
        self.active_interactions = np.array(sorted(active_interactions), dtype=np.int)
        self.interactions = self.full_dataset.interactions[self.active_interactions, :]

        if self.active_objects.size < self.full_dataset.num_objects:
            inactive_objects = set(range(self.full_dataset.num_objects)) - set(self.active_objects.tolist())
            new_bg_inds = np.array([i for i, l in enumerate(self.pc_box_labels) if l in inactive_objects])
            self.pc_box_labels[new_bg_inds] = -1
        if len(self.active_actions) < self.full_dataset.num_actions:
            assert self.split != Splits.TEST
            inactive_actions = np.array(set(range(self.full_dataset.num_actions)) - set(self.active_actions.tolist()))
            self.pc_action_labels[:, inactive_actions] = 0


        #############################################################################################################################
        # Cache
        #############################################################################################################################

        if not cfg.save_memory:
            self.pc_boxes_feats = PrecomputedFilesHandler.get(self.precomputed_feats_fn, 'box_feats', load_in_memory=True)
            self.pc_union_boxes_feats = PrecomputedFilesHandler.get(self.precomputed_feats_fn, 'union_boxes_feats', load_in_memory=True)

        self.pc_im_idx_to_im_idx = {}
        for pc_im_idx, pc_im_id in enumerate(self.pc_image_ids):
            im_idx = np.flatnonzero(self.image_ids == pc_im_id).tolist()  # type: List
            if im_idx:
                assert len(im_idx) == 1, im_idx
                self.pc_im_idx_to_im_idx[pc_im_idx] = im_idx[0]

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

        boxes_ext = self.pc_boxes_ext[all_box_inds, :].copy()
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

    @classmethod
    def get_splits(cls, act_inds=None, obj_inds=None):
        splits = {}
        full_dataset = HicoDet()

        # Split train/val if needed
        if cfg.val_ratio > 0:
            raise ValueError('Check that no images are actually filtered by precision thresholding, otherwise this has to be changed.')
            imgs_inds = np.random.permutation(full_dataset.split_non_empty_image_ids[Splits.TRAIN])
            num_imgs = len(full_dataset.split_data[Splits.TRAIN])
            num_train_imgs = num_imgs - int(num_imgs * cfg.val_ratio)
            splits[Splits.TRAIN] = cls(split=Splits.TRAIN, full_dataset=full_dataset, image_inds=sorted(imgs_inds.tolist()[:num_train_imgs]),
                                       object_inds=obj_inds, action_inds=act_inds)
            splits[Splits.VAL] = cls(split=Splits.TRAIN, full_dataset=full_dataset, image_inds=sorted(imgs_inds.tolist()[num_train_imgs:]),
                                     object_inds=obj_inds, action_inds=act_inds)
        else:
            splits[Splits.TRAIN] = cls(split=Splits.TRAIN, full_dataset=full_dataset, object_inds=obj_inds, action_inds=act_inds)

        tr = splits[Splits.TRAIN]
        if obj_inds is not None:
            print(f'{Splits.TRAIN.value.capitalize()} objects ({tr.active_objects.size}):', tr.active_objects.tolist())
        if act_inds is not None:
            print(f'{Splits.TRAIN.value.capitalize()} actions ({tr.active_actions.size}):', tr.active_actions.tolist())
        if obj_inds is not None or act_inds is not None:
            print(f'{Splits.TRAIN.value.capitalize()} interactions ({tr.active_interactions.size}):', tr.active_interactions.tolist())

        return splits


class BalancedTripletMLSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: HicoDetSingleHOIsSplit, hoi_batch_size, drop_last, shuffle):
        super().__init__(dataset)
        if not drop_last:
            raise NotImplementedError()

        self.batch_size = hoi_batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.dataset = dataset

        if cfg.null_as_bg:
            pos_hois_mask = np.any(dataset.pc_action_labels[:, 1:], axis=1)
            neg_hois_mask = (dataset.pc_action_labels[:, 0] > 0)
        else:
            pos_hois_mask = np.any(dataset.pc_action_labels, axis=1)
            neg_hois_mask = np.all(dataset.pc_action_labels == 0, axis=1)
        assert np.all(pos_hois_mask ^ neg_hois_mask)

        pc_ho_im_ids = dataset.pc_image_ids[dataset.pc_ho_im_idxs]
        split_ids_mask = np.zeros(max(np.max(pc_ho_im_ids), np.max(dataset.image_ids)) + 1, dtype=bool)
        split_ids_mask[dataset.image_ids] = True
        split_mask = split_ids_mask[pc_ho_im_ids]

        pos_hois_mask = pos_hois_mask & split_mask
        self.pos_samples = np.flatnonzero(pos_hois_mask)

        neg_hois_mask = neg_hois_mask & split_mask
        self.neg_samples = np.flatnonzero(neg_hois_mask)

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
