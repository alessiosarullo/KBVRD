import numpy as np
import torch.utils.data
from typing import Union

from lib.dataset.hicodet.hicodet import HicoDet
from lib.dataset.hicodet.pc_hicodet_split import PrecomputedHicoDetSplit, PrecomputedMinibatch, PrecomputedExample
from lib.dataset.hicodet.hicodet_img_split import Example
from lib.dataset.utils import Splits
from lib.utils import Timer


class PrecomputedHicoDetImgSplit(PrecomputedHicoDetSplit):
    def __init__(self, split, full_dataset: HicoDet):
        assert split == Splits.TEST
        super(PrecomputedHicoDetImgSplit, self).__init__(split, full_dataset, image_inds=None)

    def get_loader(self, batch_size, num_workers=0, **kwargs):
        if batch_size > 1:
            raise ValueError('Batch size has to be 1 during test.')

        data_loader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: PrecomputedMinibatch.collate(x),
            drop_last=False,
            # pin_memory=True,  # disable this in case of freezes
            **kwargs,
        )
        return data_loader

    def get_image_data(self, idx, precomputed) -> Union[PrecomputedExample, Example]:
        assert self.split == Splits.TEST
        im_data = self._data[idx]
        img_id = self.image_ids[idx]
        if precomputed:
            pc_im_idx = self.im_id_to_pc_im_idx[img_id]
            assert self.pc_image_ids[pc_im_idx] == img_id, (self.pc_image_ids[pc_im_idx], img_id)

            entry = PrecomputedExample(idx_in_split=idx, img_id=img_id, filename=im_data.filename, split=self.split)

            # Image data
            img_infos = self.pc_image_infos[pc_im_idx].copy()
            assert img_infos.shape == (3,)
            entry.img_size = img_infos[:2]
            entry.scale = img_infos[2]

            # Object data
            img_box_inds = np.flatnonzero(self.pc_box_im_idxs == pc_im_idx)  # FIXME so inefficient
            if img_box_inds.size > 0:
                start, end = img_box_inds[0], img_box_inds[-1] + 1
                assert np.all(img_box_inds == np.arange(start, end))  # slicing is much more efficient with H5 files

                entry.precomp_boxes_ext = self.pc_boxes_ext[start:end, :]
                entry.precomp_box_feats = self.pc_boxes_feats[start:end, :]
                assert self.pc_box_labels is None

                # HOI data
                img_hoi_inds = np.flatnonzero(self.pc_ho_im_idxs == pc_im_idx)  # FIXME same as above
                if img_hoi_inds.size > 0:
                    start, end = img_hoi_inds[0], img_hoi_inds[-1] + 1
                    assert np.all(img_hoi_inds == np.arange(start, end))  # slicing is much more efficient with H5 files

                    entry.precomp_hoi_infos = self.pc_ho_infos[start:end, :]
                    entry.precomp_hoi_union_boxes = self.pc_union_boxes[start:end, :]
                    entry.precomp_hoi_union_feats = self.pc_union_boxes_feats[start:end, :]
                    assert self.pc_action_labels is None
        else:
            entry = Example(idx_in_split=idx, img_id=img_id, filename=im_data.filename, split=self.split)
            entry.gt_boxes = im_data.boxes.astype(np.float, copy=False)
            entry.gt_obj_classes = im_data.box_classes.copy()
            entry.gt_hois = im_data.hois.copy()
        return entry

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx) -> PrecomputedExample:
        Timer.get('GetBatch').tic()
        entry = self.get_image_data(idx, precomputed=True)
        Timer.get('GetBatch').toc()
        return entry
