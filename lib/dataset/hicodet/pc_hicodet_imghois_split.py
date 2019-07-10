import numpy as np
import torch
import torch.utils.data

from config import cfg
from lib.dataset.hicodet.pc_hicodet_split import PrecomputedHicoDetSplit, PrecomputedMinibatch, PrecomputedExample
from lib.dataset.utils import Splits
from lib.stats.utils import Timer


class PrecomputedHicoDetImgHOISplit(PrecomputedHicoDetSplit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.split == Splits.TEST:
            raise ValueError('HOI-oriented dataset can only be used during training (labels are required to balance examples).')

    def get_loader(self, batch_size, num_workers=0, num_gpus=1, shuffle=None, drop_last=True, **kwargs):
        if shuffle is None:
            shuffle = True if self.split == Splits.TRAIN else False
        batch_size = batch_size * num_gpus

        data_loader = torch.utils.data.DataLoader(
            dataset=self,
            batch_sampler=BalancedImgSampler(self, batch_size, drop_last, shuffle),
            num_workers=num_workers,
            collate_fn=lambda x: PrecomputedMinibatch.collate(x),
            # pin_memory=True,  # disable this in case of freezes
            **kwargs,
        )
        return data_loader

    def __getitem__(self, idx) -> PrecomputedExample:
        Timer.get('GetBatch').tic()
        im_idx, fg_inds, bg_inds = idx

        im_data = self._data[im_idx]
        img_id = self.image_ids[im_idx]
        pc_im_idx = self.im_id_to_pc_im_idx[img_id]
        assert self.pc_image_ids[pc_im_idx] == img_id, (self.pc_image_ids[pc_im_idx], img_id)

        entry = PrecomputedExample(idx_in_split=im_idx, img_id=self.image_ids[im_idx], filename=im_data.filename, split=self.split)

        # Image data
        img_infos = self.pc_image_infos[pc_im_idx].copy()
        assert img_infos.shape == (3,)
        entry.img_size = img_infos[:2]
        entry.scale = img_infos[2]

        # Object data
        img_box_inds = np.flatnonzero(self.pc_box_im_idxs == pc_im_idx)
        if img_box_inds.size > 0:
            box_start, box_end = img_box_inds[0], img_box_inds[-1] + 1
            assert np.all(img_box_inds == np.arange(box_start, box_end))  # slicing is much more efficient with H5 files

            precomp_boxes_ext = self.pc_boxes_ext[box_start:box_end, :]
            precomp_box_feats = self.pc_boxes_feats[box_start:box_end, :]
            precomp_box_labels = self.pc_box_labels[box_start:box_end].copy()

            # HOI data
            img_hoi_inds = np.flatnonzero(self.pc_ho_im_idxs == pc_im_idx)
            if img_hoi_inds.size > 0:
                hoi_start, hoi_end = img_hoi_inds[0], img_hoi_inds[-1] + 1
                assert np.all(img_hoi_inds == np.arange(hoi_start, hoi_end))  # slicing is much more efficient with H5 files
                precomp_hoi_infos = self.pc_ho_infos[hoi_start:hoi_end, :].copy()
                precomp_hoi_union_boxes = self.pc_union_boxes[hoi_start:hoi_end, :]
                precomp_hoi_union_feats = self.pc_union_boxes_feats[hoi_start:hoi_end, :]
                precomp_action_labels = self.pc_action_labels[hoi_start:hoi_end, :]

                # Filter according to sample
                assert isinstance(fg_inds, list) and isinstance(bg_inds, list)
                inds = np.array(fg_inds + bg_inds) - hoi_start
                precomp_hoi_infos = precomp_hoi_infos[inds, :]
                precomp_hoi_union_boxes = precomp_hoi_union_boxes[inds, :]
                precomp_hoi_union_feats = precomp_hoi_union_feats[inds, :]
                precomp_action_labels = precomp_action_labels[inds, :]

                assert np.all(precomp_hoi_infos >= 0)
                entry.precomp_action_labels = precomp_action_labels
                entry.precomp_hoi_infos = precomp_hoi_infos
                entry.precomp_hoi_union_boxes = precomp_hoi_union_boxes
                entry.precomp_hoi_union_feats = precomp_hoi_union_feats
            else:
                assert self.split == Splits.TEST, (im_idx, pc_im_idx, img_id, im_data.filename)

            entry.precomp_boxes_ext = precomp_boxes_ext
            entry.precomp_box_feats = precomp_box_feats
            entry.precomp_box_labels = precomp_box_labels
        assert (entry.precomp_box_labels is None and entry.precomp_action_labels is None) or \
               (entry.precomp_box_labels is not None and entry.precomp_action_labels is not None)
        Timer.get('GetBatch').toc()
        return entry


class BalancedImgSampler(torch.utils.data.BatchSampler):
    def __init__(self, dataset: PrecomputedHicoDetImgHOISplit, hoi_batch_size, drop_last, shuffle):
        sampler = torch.utils.data.RandomSampler(dataset) if shuffle else torch.utils.data.SequentialSampler(dataset)
        super().__init__(sampler, hoi_batch_size, drop_last)
        if not drop_last:
            raise NotImplementedError()

        self.hoi_batch_size = self.batch_size
        self.dataset = dataset

        raise NotImplementedError()  # the following has to be fixed or deleted altogether
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
        self.split_mask = np.array([dataset.pc_image_ids[im_ind] in image_ids for im_ind in dataset.pc_ho_im_idxs])
        self.pos_hois_mask = pos_hois_mask & self.split_mask
        self.neg_hois_mask = neg_hois_mask & self.split_mask

        self.hoi_range_per_pc_img_idx = {}
        for i, im_idx in enumerate(dataset.pc_ho_im_idxs):
            start, end = self.hoi_range_per_pc_img_idx.setdefault(im_idx, [i, -1])
            self.hoi_range_per_pc_img_idx[im_idx][1] = i

        # Check
        for im_idx in np.unique(dataset.pc_ho_im_idxs):
            start, end = self.hoi_range_per_pc_img_idx[im_idx]
            fg_hois_inds = np.flatnonzero(dataset.pc_action_labels[start:end + 1, 0] == 0)
            bg_hois_inds = np.flatnonzero(dataset.pc_action_labels[start:end + 1, 0] > 0)
            if bg_hois_inds.size > 0:
                assert np.all(fg_hois_inds == np.arange(bg_hois_inds[0]))  # all positives, then all negatives

        self.neg_pos_ratio = cfg.opt.hoi_bg_ratio
        self.pos_per_batch = hoi_batch_size / (self.neg_pos_ratio + 1)
        assert self.pos_per_batch == int(self.pos_per_batch)
        self.pos_per_batch = int(self.pos_per_batch)
        self.neg_per_batch = hoi_batch_size - self.pos_per_batch

        self.batches = self.get_all_batches()

    def __iter__(self):
        for batch in self.batches:
            yield batch
        self.batches = self.get_all_batches()

    def get_all_batches(self):
        batches = []
        pos_inds, neg_inds_dict = [], {}
        pc_im_idx_to_im_idx = {}
        for im_idx in self.sampler:
            pc_im_idx = self.dataset.im_id_to_pc_im_idx[self.dataset.image_ids[im_idx]]
            assert pc_im_idx not in pc_im_idx_to_im_idx
            pc_im_idx_to_im_idx[pc_im_idx] = im_idx

            img_hois_start_idx, img_hois_end_idx = self.hoi_range_per_pc_img_idx[pc_im_idx]
            start, end = img_hois_start_idx, img_hois_end_idx + 1
            assert np.all(self.split_mask[start:end])  # image belongs to split

            im_pos_inds = start + np.flatnonzero(self.pos_hois_mask[start:end])
            im_neg_inds = start + np.flatnonzero(self.neg_hois_mask[start:end])

            pos_inds.append(np.stack([np.full_like(im_pos_inds, fill_value=pc_im_idx), im_pos_inds], axis=1))
            neg_inds_dict[pc_im_idx] = im_neg_inds
        pos_inds = np.concatenate(pos_inds, axis=0)

        assert self.drop_last
        num_batches = int(pos_inds.shape[0] // self.pos_per_batch)

        for i in range(num_batches):
            start, end = i * self.pos_per_batch, (i + 1) * self.pos_per_batch
            pos_pc_im_idxs = pos_inds[start:end, 0]
            pos_hoi_idxs = pos_inds[start:end, 1]
            u_pc_im_idxs, num_pos_per_img = np.unique(pos_pc_im_idxs, return_counts=True)

            batch_neg_inds = []
            for pc_im_idx, num_im_pos in zip(u_pc_im_idxs, num_pos_per_img):
                neg_inds_sample = np.random.permutation(neg_inds_dict[pc_im_idx])[:num_im_pos * self.neg_pos_ratio]
                batch_neg_inds.append(neg_inds_sample)
            batch_neg_inds = np.concatenate(batch_neg_inds)
            if batch_neg_inds.size < self.neg_per_batch:
                resampled_negs = np.random.choice(batch_neg_inds, self.neg_per_batch - batch_neg_inds.size, replace=True)
                batch_neg_inds = np.concatenate([batch_neg_inds, resampled_negs])

            data_per_im = {}
            for pc_im_idx, pos_hoi_idx in zip(pos_pc_im_idxs, pos_hoi_idxs):
                data_per_im.setdefault(pc_im_idx, {'pos': [], 'neg': []})['pos'].append(pos_hoi_idx)
            for neg_hoi_idx in batch_neg_inds:
                data_per_im[self.dataset.pc_ho_im_idxs[neg_hoi_idx]]['neg'].append(neg_hoi_idx)

            batch = []
            for pc_im_idx, v in data_per_im.items():
                batch.append([pc_im_idx_to_im_idx[pc_im_idx], v['pos'], v['neg']])
            assert sum([len(x[1]) for x in batch]) == self.pos_per_batch
            assert sum([len(x[2]) for x in batch]) == self.neg_per_batch

            batches.append(batch)

        return batches

    def __len__(self):
        return len(self.batches)
