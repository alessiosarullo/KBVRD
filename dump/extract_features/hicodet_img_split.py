import os
from typing import List

import cv2
import numpy as np

from config import cfg
from lib.dataset.hicodet.hicodet import HicoDet, HicoDetImData
from lib.dataset.utils import Splits, get_hico_to_coco_mapping, Example


class HicoDetImgSplit:
    def __init__(self, split, full_dataset: HicoDet):
        assert split in Splits and split != Splits.VAL

        self.split = split
        self.full_dataset = full_dataset  # type: HicoDet

        # Initilise data (possibly filter).
        split_data = self.full_dataset.split_data[split]  # type: List[HicoDetImData]
        image_ids = list(range(len(split_data)))
        if split == Splits.TRAIN:
            im_with_interactions = self.full_dataset.split_non_empty_image_ids[Splits.TRAIN]
            num_old_images, num_new_images = len(image_ids), len(im_with_interactions)
            if num_new_images < num_old_images:
                print(f'Images have been discarded due to not having objects'
                      f'{" or only having background interactions" if cfg.filter_bg_only else ""}. '
                      f'Image index has changed (from {num_old_images} images to {num_new_images}).')
                image_ids = [image_ids[i] for i in im_with_interactions]
                split_data = [split_data[i] for i in im_with_interactions]
            assert len(split_data) == len(image_ids)
            assert image_ids == sorted(image_ids)
        self.image_ids = image_ids
        self._data = split_data

        # Compute mappings to COCO
        self.hico_to_coco_mapping = get_hico_to_coco_mapping(hico_objects=self.full_dataset.objects)

    @property
    def human_class(self):
        return self.full_dataset.human_class

    @property
    def num_objects(self):
        return self.full_dataset.num_objects

    @property
    def num_actions(self):
        return self.full_dataset.num_actions

    @property
    def num_interactions(self):
        return self.full_dataset.num_interactions

    @property
    def num_images(self):
        return len(self.image_ids)

    @property
    def img_dir(self):
        return self.full_dataset.get_img_dir(self.split)

    def get_img_entry(self, idx, read_img=True) -> Example:
        im_data = self._data[idx]

        entry = Example(idx_in_split=idx, img_id=self.image_ids[idx], filename=im_data.filename, split=self.split)
        entry.gt_boxes = im_data.boxes.astype(np.float, copy=False)
        entry.gt_obj_classes = im_data.box_classes.copy()
        entry.gt_hois = im_data.hois.copy()
        if read_img:
            entry.image = cv2.imread(os.path.join(self.img_dir, im_data.filename))
        return entry

    def __getitem__(self, idx):
        return self.get_img_entry(idx)

    def __len__(self):
        return self.num_images
