import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

from config import Configs as cfg
from scripts.utils import Timer
from lib.dataset.hicodet_driver import HicoDet as HicoDetDriver
from lib.dataset.utils import Splits, preprocess_img
from lib.containers import Minibatch, Example


class HicoDetSplit(Dataset):
    def __init__(self, split, im_inds=None, filter_invisible=True, hicodet_driver=None, flipping_prob=0):
        """
        """
        assert split in Splits
        hicodet_driver = hicodet_driver or HicoDetDriver()

        self.split = split
        self.hicodet = hicodet_driver
        self.flipping_prob = flipping_prob
        print('Flipping is %s.' % (('enabled with probability %.2f' % flipping_prob) if flipping_prob > 0 else 'disabled'))

        # Initialize
        self.annotations = hicodet_driver.split_data[split]['annotations']
        if im_inds is not None:
            self.annotations = [self.annotations[i] for i in im_inds]
            self.image_ids = im_inds
        else:
            self.image_ids = list(range(len(self.annotations)))
        if self.is_train and filter_invisible:
            vis_im_inds, self.annotations = zip(*[(i, ann) for i, ann in enumerate(self.annotations)
                                                  if any([not inter['invis'] for inter in ann['interactions']])])
            num_old_images, num_new_images = len(self.image_ids), len(vis_im_inds)
            if num_new_images < num_old_images:
                print('Some images have been discarded due to not having visible interactions. '
                      'Image index has changed (from %d images to %d).' % (num_old_images, num_new_images))
                self.image_ids = [self.image_ids[i] for i in vis_im_inds]

        self._im_boxes, self._im_box_classes, self._im_inters = self.compute_gt_data()
        assert len(self._im_boxes) == len(self._im_box_classes) == len(self._im_inters) == len(self.annotations) == len(self.image_ids)

        # You could add data augmentation here.
        pass

        # In case of precomputed features
        if cfg.program.load_precomputed_feats:
            assert self.flipping_prob == 0  # TODO extract features for flipped image?
            precomputed_feats_fn = cfg.program.precomputed_feats_file_format % cfg.model.rcnn_arch
            self.pc_feats_file = h5py.File(precomputed_feats_fn, 'r')
            self.pc_box_pred_classes = self.pc_feats_file['box_pred_classes'][:]
            try:
                self.pc_box_im_inds = self.pc_feats_file['box_im_ids'][:]
                self.pc_image_ids = self.pc_feats_file['image_index'][:]
            except KeyError:  # Old names
                self.pc_box_im_inds = self.pc_feats_file['box_im_inds'][:]
                self.pc_image_ids = self.pc_feats_file['image_ids'][:]

            # TODO decide whether to add support when this is not true
            assert len(set(self.image_ids) - set(self.pc_image_ids.tolist())) == 0
            assert len(self.pc_image_ids) == len(set(self.pc_image_ids))
            self.im_id_to_pc_im_idx = {}
            for im_id in self.image_ids:
                pc_im_idx = np.flatnonzero(self.pc_image_ids == im_id).tolist()
                assert len(pc_im_idx) == 1, pc_im_idx
                assert im_id not in self.im_id_to_pc_im_idx
                self.im_id_to_pc_im_idx[im_id] = pc_im_idx[0]
        else:
            self.pc_feats_file = None

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
            for interaction in img_ann['interactions']:
                if not interaction['invis']:
                    curr_num_hum_boxes = int(sum([b.shape[0] for b in im_hum_boxes]))
                    curr_num_obj_boxes = int(sum([b.shape[0] for b in im_obj_boxes]))

                    # Interaction
                    pred_class = hd.get_predicate_index(interaction['id'])
                    new_inters = interaction['conn']
                    new_inters = np.stack([new_inters[:, 0] + curr_num_hum_boxes,
                                           np.full(new_inters.shape[0], fill_value=pred_class, dtype=np.int),
                                           new_inters[:, 1] + curr_num_obj_boxes
                                           ], axis=1)
                    im_interactions.append(new_inters)

                    # Human
                    im_hum_boxes.append(interaction['hum_bbox'])

                    # Object
                    obj_boxes = interaction['obj_bbox']
                    im_obj_boxes.append(obj_boxes)
                    obj_class = hd.get_object_index(interaction['id'])
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

    def get_loader(self, batch_size, num_workers=0, num_gpus=1, shuffle=None, drop_last=True, **kwargs):
        if shuffle is None:
            shuffle = True if self.is_train else False
        data_loader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size * num_gpus,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda x: Minibatch.collate(x),
            drop_last=drop_last,
            # pin_memory=True,  # disable this in case of freezes
            **kwargs,
        )
        return data_loader

    def __getitem__(self, idx):
        Timer.get('Epoch', 'GetBatch').tic()

        # Read the image
        img_fn = self.annotations[idx]['file']
        img_id = self.image_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_fn))
        img_h, img_w = img.shape[:2]
        gt_boxes = self._im_boxes[idx].astype(np.float, copy=False)

        # Optionally flip the image if we're doing training
        flipped = self.is_train and np.random.random() < self.flipping_prob
        if flipped:
            img = img[:, ::-1, :]  # NOTE: change this to [:, :, ::-1] if the image is read through PIL
            gt_boxes[:, [0, 2]] = img_w - gt_boxes[:, [2, 0]]

        preprocessed_im, img_scale_factor = preprocess_img(img)
        entry = Example(idx_in_split=idx, img_id=img_id, img_fn=img_fn,
                        image=preprocessed_im, img_size=(img_h, img_w), img_scale_factor=img_scale_factor, flipped=flipped,
                        gt_boxes=gt_boxes, gt_obj_classes=self._im_box_classes[idx].copy(), gt_hois=self._im_inters[idx].copy())

        if self.pc_feats_file is not None:
            pc_im_idx = self.im_id_to_pc_im_idx[img_id]
            assert self.pc_image_ids[pc_im_idx] == img_id, (self.pc_image_ids[pc_im_idx], img_id)
            inds = np.flatnonzero(self.pc_box_im_inds == pc_im_idx)
            start, end = inds[0], inds[-1] + 1
            assert np.all(inds == np.arange(start, end))  # slicing is much more efficient with H5 files
            entry.precomputed_boxes = self.pc_feats_file['boxes'][start:end, :]
            entry.precomputed_obj_scores = self.pc_feats_file['box_scores'][start:end, :]
            entry.precomputed_box_feats = self.pc_feats_file['box_feats'][start:end, :]
            entry.precomputed_obj_classes = self.pc_box_pred_classes[start:end, :]
        Timer.get('Epoch', 'GetBatch').toc()
        return entry

    def __len__(self):
        return len(self.image_ids)


def main():
    hds = HicoDetSplit(Splits.TRAIN, im_inds=[12, 13, 14])


if __name__ == '__main__':
    main()
