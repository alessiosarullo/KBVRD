"""
File that involves dataloaders for the Visual Genome dataset.
"""

import json
import os
import random
# from dataloaders.blob import Blob
# from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from collections import defaultdict

import h5py
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from drivers.hicodet_driver import HicoDet as HicoDetDriver
from utils.data import Splits

# FIXME normalisation values
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IM_SCALE = 480


class SquarePad(object):
    def __call__(self, img):
        w, h = img.size
        img_padded = ImageOps.expand(img, border=(0, 0, max(h - w, 0), max(w - h, 0)),
                                     fill=(int(NORM_MEAN[0] * 256), int(NORM_MEAN[1] * 256), int(NORM_MEAN[2] * 256)))
        return img_padded


class HicoDetSplit(Dataset):
    def __init__(self, driver: HicoDetDriver, split, im_inds=None, filter_invisible=True):
        """
        """
        assert split in Splits
        self.split = split
        self.driver = driver
        self.image_index = im_inds

        # Initialize
        self.annotations = driver.split_data[split]['annotations']
        if im_inds is not None:
            self.annotations = [self.annotations[i] for i in im_inds]
        if filter_invisible:
            vis_im_inds, self.annotations = zip(*[(i, ann) for i, ann in enumerate(self.annotations)
                                                  if any([not inter['invis'] for inter in ann['interactions']])])
            if im_inds is not None:
                self.image_index = [im_inds[i] for i in vis_im_inds]
            print('Some images have been discarded due to not having visible interactions. Image index has changed.')

        self._im_boxes, self._im_box_classes, self._im_inters = self.compute_gt_data()
        assert len(self._im_boxes) == len(self._im_box_classes) == len(self._im_inters) == len(self.annotations) == len(self.image_index)

        # You could add data augmentation here.
        pass

        # Image transformation pipeline
        tform = [
            SquarePad,
            Resize(IM_SCALE),
            ToTensor(),
            Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]
        self.transform_pipeline = Compose(tform)

    def compute_gt_data(self):
        im_without_visible_interactions = []
        boxes, box_classes, interactions = [], [], []
        for i, img_ann in enumerate(self.annotations):
            im_hum_boxes, im_obj_boxes, im_obj_box_classes, im_interactions = [], [], [], []
            for inter in img_ann['interactions']:
                if not inter['invis']:
                    curr_num_hum_boxes = sum([b.shape[0] for b in im_hum_boxes])
                    curr_num_obj_boxes = sum([b.shape[0] for b in im_obj_boxes])
                    inters = inter['conn']
                    pred_class = self.driver.interactions[inter['id']]['pred']
                    inters = np.stack([inters[:, 0], np.ones(inters.shape[0], dtype=np.int) * pred_class, inters[:, 1]], axis=1)
                    im_interactions.append(inters + np.array([[curr_num_hum_boxes, 0, curr_num_obj_boxes]], dtype=np.int))

                    im_hum_boxes.append(inter['hum_bbox'])

                    obj_boxes = inter['obj_bbox']
                    obj_class = self.driver.obj_class_index[self.driver.interactions[inter['id']]['obj']]
                    im_obj_boxes.append(obj_boxes)
                    im_obj_box_classes.append(np.ones(obj_boxes.shape[0], dtype=np.int) * obj_class)

            if im_hum_boxes:
                assert im_obj_boxes
                assert im_obj_box_classes
                assert im_interactions
                im_hum_boxes, inv_ind = np.unique(np.concatenate(im_hum_boxes, axis=0), axis=0, return_inverse=True)
                num_hum_boxes = im_hum_boxes.shape[0]

                im_obj_boxes = np.concatenate(im_obj_boxes)
                im_obj_box_classes = np.concatenate(im_obj_box_classes)

                im_interactions = np.concatenate(im_interactions)
                im_interactions[:, 0] = np.array([inv_ind[h] for h in im_interactions[:, 0]], dtype=np.int)
                im_interactions[:, 2] += num_hum_boxes

                boxes.append(np.concatenate([im_hum_boxes, im_obj_boxes], axis=0))
                box_classes.append(np.concatenate([-np.ones(num_hum_boxes, dtype=np.int), im_obj_box_classes]))  # humans have class -1
                interactions.append(im_interactions)
            else:
                boxes.append([])
                box_classes.append([])
                interactions.append([])
                im_without_visible_interactions.append(i)
        return boxes, box_classes, interactions

    @property
    def is_train(self):
        return self.split == Splits.TRAIN

    def __getitem__(self, index):
        # Read the image
        img_fn = self.annotations[index]['file']
        image_unpadded = Image.open(os.path.join(self.driver.split_data[self.split]['img_dir'], img_fn)).convert('RGB')
        img_w, img_h = image_unpadded.size
        assert (img_w, img_h) == self.annotations[index]['img_size'][:2], ((img_w, img_h), self.annotations[index]['img_size'][:2])

        # Compute rescaling factor
        img_scale_factor = IM_SCALE / max(img_w, img_h)
        if img_h > img_w:
            im_size = (IM_SCALE, int(img_w * img_scale_factor), image_unpadded.size[2])
        elif img_h < img_w:
            im_size = (int(img_h * img_scale_factor), IM_SCALE, image_unpadded.size[2])
        else:
            im_size = (IM_SCALE, IM_SCALE, image_unpadded.size[2])

        # Optionally flip the image if we're doing training
        gt_boxes = self._im_boxes[index].copy()
        flipped = self.is_train and np.random.random() > 0.5
        if flipped:
            image_unpadded = image_unpadded.transpose(Image.FLIP_LEFT_RIGHT)
            gt_boxes[:, [0, 2]] = img_w - gt_boxes[:, [2, 0]]

        entry = {
            'index': index,
            'fn': img_fn,
            'img': self.transform_pipeline(image_unpadded),
            'img_size': im_size,
            'scale': img_scale_factor,  # Multiply the boxes by this.
            'gt_boxes': gt_boxes,
            'gt_box_classes': self._im_box_classes[index].copy(),
            'gt_inters': self._im_inters[index].copy(),
            'flipped': flipped,
        }
        return entry

    def __len__(self):
        return len(self.image_index)

    @property
    def num_predicates(self):
        return len(self.driver.predicates)

    @property
    def num_object_classes(self):
        return len(self.driver.objects)


class VGDataLoader(torch.utils.data.DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """

    @staticmethod
    def vg_collate(data, num_gpus=3, is_train=False, mode='det'):
        assert mode in ('det', 'rel')
        blob = Blob(mode=mode, is_train=is_train, num_gpus=num_gpus, batch_size_per_gpu=len(data) // num_gpus)
        for d in data:
            blob.append(d)
        blob.reduce()
        return blob

    @classmethod
    def get_split(cls, dataset, is_train, batch_size=3, num_workers=1, num_gpus=3, mode='det',
                  **kwargs):
        assert mode in ('det', 'rel')
        if is_train:
            data_loader = cls(
                dataset=dataset,
                batch_size=batch_size * num_gpus,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=lambda x: cls.vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=True),
                drop_last=True,
                # pin_memory=True,
                **kwargs,
            )
        else:
            data_loader = cls(
                dataset=dataset,
                batch_size=batch_size * num_gpus if mode == 'det' else num_gpus,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=lambda x: cls.vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=False),
                drop_last=True,
                # pin_memory=True,
                **kwargs,
            )
        return data_loader


def main():
    hds = HicoDetSplit(HicoDetDriver(), Splits.TRAIN, im_inds=[12, 13, 14])


if __name__ == '__main__':
    main()
