from typing import List

import h5py
import numpy as np
import torch
import torch.utils.data

from config import cfg
from lib.dataset.hicodet.hicodet_split import HicoDetSplitBuilder, HicoDetSplit, remap_box_pairs
from lib.dataset.utils import Splits
from lib.stats.utils import Timer


class PrecomputedExample:
    def __init__(self, idx_in_split, img_id, filename, split):
        self.index = idx_in_split
        self.id = img_id
        self.filename = filename
        self.split = split

        self.img_size = None
        self.scale = None

        self.precomp_boxes_ext = None
        self.precomp_box_feats = None

        self.precomp_hoi_infos = None
        self.precomp_hoi_union_boxes = None
        self.precomp_hoi_union_feats = None

        self.precomp_box_labels = None
        self.precomp_action_labels = None


class PrecomputedMinibatch:
    def __init__(self):
        self.img_infos = []
        self.other_ex_data = []
        self.pc_boxes_ext = []
        self.pc_box_feats = []
        self.pc_ho_infos = []
        self.pc_ho_union_boxes = []
        self.pc_ho_union_feats = []
        self.pc_box_labels = []
        self.pc_action_labels = []

        self.epoch = None
        self.iter = None

    def append(self, ex: PrecomputedExample):
        im_id_in_batch = len(self.img_infos)
        self.img_infos += [np.array([*ex.img_size, ex.scale], dtype=np.float32)]

        self.other_ex_data += [{'index': ex.index,
                                'id': ex.id,
                                'fn': ex.filename,
                                'split': ex.split,
                                'im_size': ex.img_size,  # this is the original one and won't be changed
                                'im_scale': ex.scale,
                                }]

        boxes_ext = ex.precomp_boxes_ext
        if boxes_ext is not None:
            boxes_ext[:, 0] = im_id_in_batch
            self.pc_boxes_ext += [boxes_ext]
            self.pc_box_feats += [ex.precomp_box_feats]

            self.pc_box_labels += [ex.precomp_box_labels]

            hoi_infos = ex.precomp_hoi_infos
            if hoi_infos is not None:
                num_boxes = sum([boxes.shape[0] for boxes in self.pc_boxes_ext[:-1]])
                hoi_infos[:, 0] = im_id_in_batch
                hoi_infos[:, 1:] += num_boxes
                self.pc_ho_infos += [hoi_infos]
                self.pc_ho_union_boxes += [ex.precomp_hoi_union_boxes]
                self.pc_ho_union_feats += [ex.precomp_hoi_union_feats]

                self.pc_action_labels += [ex.precomp_action_labels]

    def vectorize(self, device):
        for k, v in self.__dict__.items():
            if k.startswith('pc_') and ('label' not in k):
                if not v:
                    v = [np.empty(0)]
                self.__dict__[k] = np.concatenate(v, axis=0)

        assert self.pc_boxes_ext.shape[0] == self.pc_box_feats.shape[0]
        assert self.pc_ho_infos.shape[0] == self.pc_ho_union_boxes.shape[0]

        img_infos = np.stack(self.img_infos, axis=0)
        img_infos[:, 0] = max(img_infos[:, 0])
        img_infos[:, 1] = max(img_infos[:, 1])
        self.img_infos = torch.tensor(img_infos, dtype=torch.float32, device=device)

        if self.pc_box_labels[0] is None:
            assert all([l is None for l in self.pc_box_labels])
            assert all([l is None for l in self.pc_action_labels])
            self.pc_box_labels = self.pc_action_labels = None
        else:
            assert all([l is not None for l in self.pc_box_labels])
            assert all([l is not None for l in self.pc_action_labels])
            assert len(self.pc_box_labels) == len(self.pc_action_labels) == self.img_infos.shape[0], \
                (len(self.pc_box_labels), len(self.pc_action_labels), self.img_infos.shape[0])
            self.pc_box_labels = np.concatenate(self.pc_box_labels, axis=0)
            self.pc_action_labels = np.concatenate(self.pc_action_labels, axis=0)
            assert self.pc_boxes_ext.shape[0] == self.pc_box_labels.shape[0]
            assert self.pc_ho_infos.shape[0] == self.pc_action_labels.shape[0]

    @classmethod
    def collate(cls, examples, **kwargs):
        minibatch = cls()
        for ex in examples:
            minibatch.append(ex)
        minibatch.vectorize(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return minibatch


class PrecomputedHicoDetSplit(HicoDetSplit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        precomputed_feats_fn = cfg.program.precomputed_feats_file_format % (cfg.model.rcnn_arch,
                                                                            (Splits.TEST if self.split == Splits.TEST else Splits.TRAIN).value)
        print('Loading precomputed feats for %s split from %s.' % (self.split.value, precomputed_feats_fn))
        pc_feats_file = h5py.File(precomputed_feats_fn, 'r')
        assert pc_feats_file['box_feats'].shape[1] == 2048  # FIXME magic constant

        self.pc_image_ids = pc_feats_file['image_ids'][:]
        self.pc_image_infos = pc_feats_file['img_infos'][:]

        self.pc_boxes_ext = pc_feats_file['boxes_ext'][:]
        self.pc_boxes_feats = pc_feats_file['box_feats']
        self.pc_union_boxes_feats = pc_feats_file['union_boxes_feats']
        try:
            self.pc_box_labels = pc_feats_file['box_labels'][:]
        except KeyError:
            self.pc_box_labels = None

        self.pc_ho_infos = pc_feats_file['ho_infos'][:].astype(np.int)
        self.pc_union_boxes = pc_feats_file['union_boxes'][:]
        try:
            self.pc_action_labels = pc_feats_file['action_labels'][:]
        except KeyError:
            self.pc_action_labels = None

        # Filter HOIs. Object class filtering is currently not supported.
        if self.active_object_classes.size < self.hicodet.num_object_classes:
            raise NotImplementedError('Object class filtering is not supported.')
        if len(self.active_predicates) < self.hicodet.num_predicates and self.split != Splits.TEST:
            self.pc_action_labels = self.pc_action_labels[:, self.active_predicates]
            self.pc_hoi_mask = np.any(self.pc_action_labels, axis=1)
            # Note: boxes with no interactions are NOT filtered
        else:
            self.pc_hoi_mask = None

        # Derived
        self.pc_box_im_idxs = self.pc_boxes_ext[:, 0].astype(np.int)
        self.pc_ho_im_idxs = self.pc_ho_infos[:, 0]

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
        return self.pc_boxes_feats.shape[1]

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

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx) -> PrecomputedExample:
        Timer.get('GetBatch').tic()
        im_data = self._data[idx]
        img_id = self.image_ids[idx]
        pc_im_idx = self.im_id_to_pc_im_idx[img_id]
        assert self.pc_image_ids[pc_im_idx] == img_id, (self.pc_image_ids[pc_im_idx], img_id)

        entry = PrecomputedExample(idx_in_split=idx, img_id=img_id, filename=im_data.filename, split=self.split)

        # Image data
        img_infos = self.pc_image_infos[pc_im_idx].copy()
        assert img_infos.shape == (3,)
        entry.img_size = img_infos[:2]
        entry.scale = img_infos[2]

        # Object data
        img_box_inds = np.flatnonzero(self.pc_box_im_idxs == pc_im_idx)
        if img_box_inds.size > 0:
            start, end = img_box_inds[0], img_box_inds[-1] + 1
            assert np.all(img_box_inds == np.arange(start, end))  # slicing is much more efficient with H5 files

            precomp_boxes_ext = self.pc_boxes_ext[start:end, :]
            precomp_box_feats = self.pc_boxes_feats[start:end, :]
            box_inds = None
            if self.pc_box_labels is not None:
                # TODO mapping of obj inds for rels
                precomp_box_labels = self.pc_box_labels[start:end].copy()

                if self.active_object_classes.size < self.num_object_classes:
                    precomp_boxes_ext = precomp_boxes_ext[:, np.concatenate([np.arange(5), 5 + self.active_object_classes])]
                    num_boxes = precomp_box_labels.shape[0]
                    bg_box_mask = (precomp_box_labels < 0)

                    precomp_box_labels_one_hot = np.zeros([num_boxes, len(self.hicodet.objects)], dtype=precomp_box_labels.dtype)
                    precomp_box_labels_one_hot[np.arange(num_boxes), precomp_box_labels] = 1
                    precomp_box_labels_one_hot[bg_box_mask, :] = 0
                    precomp_box_labels_one_hot = precomp_box_labels_one_hot[:, self.active_object_classes]
                    feasible_box_labels_inds = np.any(precomp_box_labels_one_hot, axis=1) | bg_box_mask

                    box_inds = np.full_like(precomp_box_labels, fill_value=-1)
                    box_inds[feasible_box_labels_inds] = np.arange(np.sum(feasible_box_labels_inds))
                    precomp_boxes_ext = precomp_boxes_ext[feasible_box_labels_inds]
                    precomp_box_feats = precomp_box_feats[feasible_box_labels_inds]
                    precomp_box_labels_one_hot = precomp_box_labels_one_hot[feasible_box_labels_inds]
                    precomp_box_labels = np.argmax(precomp_box_labels_one_hot, axis=1).astype(precomp_box_labels_one_hot.dtype)
                    precomp_box_labels[~np.any(precomp_box_labels_one_hot, axis=1)] = -1
            else:
                precomp_box_labels = None

            # HOI data
            img_hoi_inds = np.flatnonzero(self.pc_ho_im_idxs == pc_im_idx)
            if img_hoi_inds.size > 0:
                start, end = img_hoi_inds[0], img_hoi_inds[-1] + 1
                assert np.all(img_hoi_inds == np.arange(start, end))  # slicing is much more efficient with H5 files
                precomp_hoi_infos = self.pc_ho_infos[start:end, :].copy()
                precomp_hoi_union_boxes = self.pc_union_boxes[start:end, :]
                precomp_hoi_union_feats = self.pc_union_boxes_feats[start:end, :]
                if self.pc_action_labels is not None:
                    precomp_action_labels = self.pc_action_labels[start:end, :]
                else:
                    precomp_action_labels = None

                if precomp_action_labels is not None:
                    assert precomp_box_labels is not None

                    # Remap HOIs box indices
                    if box_inds is not None:
                        precomp_hoi_infos[:, 1] = box_inds[precomp_hoi_infos[:, 1]]
                        precomp_hoi_infos[:, 2] = box_inds[precomp_hoi_infos[:, 2]]

                    # Filter out HOIs
                    if self.pc_hoi_mask is not None:
                        img_hoi_mask = self.pc_hoi_mask[start:end]
                        precomp_hoi_infos = precomp_hoi_infos[img_hoi_mask, :]
                        precomp_hoi_union_boxes = precomp_hoi_union_boxes[img_hoi_mask, :]
                        precomp_hoi_union_feats = precomp_hoi_union_feats[img_hoi_mask, :]
                        if precomp_action_labels is not None:
                            precomp_action_labels = precomp_action_labels[img_hoi_mask, :]

                    # Filter out HOIs
                    if False and (self.active_predicates.size < self.num_predicates or box_inds is not None):  # FIXME
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
            entry.precomp_box_labels = precomp_box_labels
        assert (entry.precomp_box_labels is None and entry.precomp_action_labels is None) or \
               (entry.precomp_box_labels is not None and entry.precomp_action_labels is not None)
        Timer.get('GetBatch').toc()
        return entry
