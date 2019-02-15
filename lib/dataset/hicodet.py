import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from lib.dataset.hicodet_driver import HicoDet as HicoDetDriver
from lib.utils.data import Splits, Minibatch, preprocess_img


class HicoDetSplit(Dataset):
    def __init__(self, split, im_inds=None, filter_invisible=True, hicodet_driver=None):
        """
        """
        assert split in Splits
        hicodet_driver = hicodet_driver or HicoDetDriver()

        self.split = split
        self.hicodet = hicodet_driver

        # Initialize
        self.annotations = hicodet_driver.split_data[split]['annotations']
        if im_inds is not None:
            self.annotations = [self.annotations[i] for i in im_inds]
            self.image_index = im_inds
        else:
            im_inds = list(range(len(self.annotations)))
        if filter_invisible:
            vis_im_inds, self.annotations = zip(*[(i, ann) for i, ann in enumerate(self.annotations)
                                                  if any([not inter['invis'] for inter in ann['interactions']])])
            self.image_index = [im_inds[i] for i in vis_im_inds]
            print('Some images have been discarded due to not having visible interactions. Image index has changed.')

        self._im_boxes, self._im_box_classes, self._im_inters = self.compute_gt_data()
        assert len(self._im_boxes) == len(self._im_box_classes) == len(self._im_inters) == len(self.annotations) == len(self.image_index)

        # You could add data augmentation here.
        pass

    @property
    def num_predicates(self):
        return len(self.hicodet.predicates)

    @property
    def num_object_classes(self):
        return len(self.hicodet.objects)

    @property
    def img_dir(self):
        return self.hicodet.get_img_dir(self.split)

    @property
    def is_train(self):
        return self.split == Splits.TRAIN

    def compute_gt_data(self):
        hd = self.hicodet
        im_without_visible_interactions = []
        boxes, box_classes, interactions = [], [], []
        for i, img_ann in enumerate(self.annotations):
            im_hum_boxes, im_obj_boxes, im_obj_box_classes, im_interactions = [], [], [], []
            for inter in img_ann['interactions']:
                if not inter['invis']:
                    curr_num_hum_boxes = sum([b.shape[0] for b in im_hum_boxes])
                    curr_num_obj_boxes = sum([b.shape[0] for b in im_obj_boxes])
                    inters = inter['conn']
                    pred_class = hd.pred_index[hd.interactions[inter['id']]['pred']]
                    inters = np.stack([inters[:, 0], np.full(inters.shape[0], fill_value=pred_class, dtype=np.int), inters[:, 1]], axis=1)
                    im_interactions.append(inters + np.array([[curr_num_hum_boxes, 0, curr_num_obj_boxes]], dtype=np.int))

                    im_hum_boxes.append(inter['hum_bbox'])

                    obj_boxes = inter['obj_bbox']
                    obj_class = hd.obj_class_index[hd.interactions[inter['id']]['obj']]
                    im_obj_boxes.append(obj_boxes)
                    im_obj_box_classes.append(np.full(obj_boxes.shape[0], fill_value=obj_class, dtype=np.int))

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
                box_classes.append(np.concatenate([np.full(num_hum_boxes, fill_value=self.hicodet.person_class, dtype=np.int), im_obj_box_classes]))
                interactions.append(im_interactions)
            else:
                boxes.append([])
                box_classes.append([])
                interactions.append([])
                im_without_visible_interactions.append(i)
        return boxes, box_classes, interactions

    def get_loader(self, batch_size, num_workers=0, num_gpus=1, **kwargs):
        data_loader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size * num_gpus,
            shuffle=True if self.is_train else False,
            num_workers=num_workers,
            collate_fn=lambda x: Minibatch.collate(x, training=self.is_train),
            drop_last=True,
            # pin_memory=True,  # disable this in case of freezes
            **kwargs,
        )
        return data_loader

    def __getitem__(self, index):
        # Read the image
        img_fn = self.annotations[index]['file']

        img = cv2.imread(os.path.join(self.img_dir, img_fn))
        img_h, img_w = img.shape[:2]
        gt_boxes = self._im_boxes[index].astype(np.float, copy=False)

        # Optionally flip the image if we're doing training
        flipped = self.is_train and np.random.random() > 0.5
        if flipped:
            img = img[:, :, ::-1]
            gt_boxes[:, [0, 2]] = img_w - gt_boxes[:, [2, 0]]

        preprocessed_im, img_scale_factor = preprocess_img(img)
        entry = {
            'index': index,
            'fn': img_fn,
            'img': preprocessed_im,
            'img_size': (img_h, img_w),
            'scale': img_scale_factor,
            'gt_boxes': gt_boxes,
            'gt_box_classes': self._im_box_classes[index].copy(),
            'gt_inters': self._im_inters[index].copy(),
            'flipped': flipped,
        }
        return entry

    def __len__(self):
        return len(self.image_index)


def main():
    hds = HicoDetSplit(Splits.TRAIN, im_inds=[12, 13, 14])


if __name__ == '__main__':
    main()
