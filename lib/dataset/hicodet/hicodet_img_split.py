import os
from typing import List

import cv2
import numpy as np

from config import cfg
from lib.dataset.hicodet.hicodet import HicoDet, HicoDetImData
from lib.dataset.utils import Splits, preprocess_img, get_hico_to_coco_mapping


class Example:
    def __init__(self, idx_in_split, img_id, filename, split):
        self.index = idx_in_split
        self.id = img_id
        self.filename = filename
        self.split = split

        self.orig_img_size = None
        self.scale = None

        self.image = None
        self.gt_boxes = None
        self.gt_obj_classes = None
        self.gt_hois = None


class HicoDetImgSplit:
    def __init__(self, split, full_dataset: HicoDet):
        assert split in Splits and split != Splits.VAL

        self.split = split
        self.full_dataset = full_dataset  # type: HicoDet

        # Initilise data (possibly filter).
        split_data = self.full_dataset.split_data[split]  # type: List[HicoDetImData]
        image_ids = list(range(len(split_data)))
        if split == Splits.TRAIN:
            im_with_interactions = []
            for i, im_data in enumerate(split_data):
                empty = im_data.boxes.size == 0
                fg_hois = np.any(im_data.hois[:, 1] != full_dataset.action_index[full_dataset.null_interaction])
                if empty:  # empty = no boxes
                    continue
                if cfg.filter_bg_only and ~fg_hois:
                    continue
                im_with_interactions.append(i)
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

        # Compute HOI triplets. Each is [human, action, object].
        hoi_triplets = []
        for im_data in self._data:
            box_classes, inters = im_data.box_classes, im_data.hois
            im_hois = np.stack([box_classes[inters[:, 0]], inters[:, 1], box_classes[inters[:, 2]]], axis=1)
            assert np.all(im_hois[:, 0] == self.full_dataset.human_class)
            hoi_triplets.append(im_hois)
        self.hoi_triplets = np.concatenate(hoi_triplets, axis=0)

        self.obj_labels = np.concatenate([im_data.box_classes for im_data in self._data])

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
        if read_img:
            raw_image = cv2.imread(os.path.join(self.img_dir, im_data.filename))
            img_h, img_w = raw_image.shape[:2]
            image, img_scale_factor = preprocess_img(raw_image)
            img_size = [img_h, img_w]

            entry.image = image
            entry.orig_img_size = img_size
            entry.scale = img_scale_factor

        entry.gt_boxes = im_data.boxes.astype(np.float, copy=False)
        entry.gt_obj_classes = im_data.box_classes.copy()
        entry.gt_hois = im_data.hois.copy()
        return entry

    def __getitem__(self, idx):
        return self.get_img_entry(idx)

    def __len__(self):
        return self.num_images
