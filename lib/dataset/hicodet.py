import os
from typing import List

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from config import Configs as cfg
from lib.containers import Minibatch, Example
from lib.dataset.hicodet_driver import HicoDet as HicoDetDriver
from lib.dataset.utils import Splits, preprocess_img
from lib.detection.wrappers import COCO_CLASSES
from scripts.utils import Timer


class HicoDetInstance(Dataset):
    def __init__(self, split, im_inds=None, pred_inds=None, obj_inds=None, filter_invisible=True, hicodet_driver=None, flipping_prob=0):
        """
        """
        # TODO docs
        assert split in Splits
        hicodet_driver = hicodet_driver or HicoDetDriver()

        # Load inds from configs first. Note that these might still be None after this step, which means all possible indices will be used.
        im_inds = im_inds or cfg.data.im_inds
        pred_inds = pred_inds or cfg.data.pred_inds
        obj_inds = obj_inds or cfg.data.obj_inds

        self.split = split
        self._hicodet = hicodet_driver
        self.flipping_prob = flipping_prob
        print('Flipping is %s.' % (('enabled with probability %.2f' % flipping_prob) if flipping_prob > 0 else 'disabled'))

        ################# Initialize
        # Set annotations and image index
        annotations = hicodet_driver.split_data[split]['annotations']
        image_ids = im_inds or list(range(len(annotations)))
        if im_inds is not None:
            annotations = [annotations[i] for i in im_inds]

        # Filter out unwanted predicates/object classes
        if obj_inds is None and pred_inds is None:
            self._objects = list(hicodet_driver.objects)
            self._predicates = list(hicodet_driver.predicates)
        else:
            obj_inds = set(obj_inds or range(len(hicodet_driver.objects)))
            pred_inds = set(pred_inds or range(len(hicodet_driver.predicates)))
            assert 0 in pred_inds
            new_im_inds, new_annotations = [], []
            for i, im_ann in enumerate(annotations):
                new_im_inters = []
                for inter in im_ann['interactions']:
                    if hicodet_driver.get_object_index(inter['id']) in obj_inds and hicodet_driver.get_predicate_index(inter['id']) in pred_inds:
                        new_im_inters.append(inter)
                if new_im_inters:
                    new_im_inds.append(i)
                    new_annotations.append({k: (v if k != 'interactions' else new_im_inters) for k, v in im_ann.items()})
            num_inters = sum([len(ann['interactions']) for ann in annotations])
            diff_num_inters = num_inters - sum([len(ann['interactions']) for ann in new_annotations])
            if diff_num_inters > 0:
                print('%d/%d interaction%s been filtered out.' % (diff_num_inters, num_inters, ' has' if diff_num_inters == 1 else 's have'))
            if len(new_im_inds) < len(image_ids):
                print('Images have been discarded due to not having feasible predicates or objects. '
                      'Image index has changed (from %d images to %d).' % (len(image_ids), len(new_im_inds)))
            annotations = new_annotations
            image_ids = [image_ids[i] for i in new_im_inds]

            # Now we can add the person class: there won't be interactions with persons as an object if not in the initial indices, but the dataset
            # includes the class anyway because the model must always be able to predict it.
            obj_inds.add(hicodet_driver.person_class)
            self._objects = [hicodet_driver.objects[i] for i in sorted(obj_inds)]
            self._predicates = [hicodet_driver.predicates[i] for i in sorted(pred_inds)]
        self._person_class_index = self._objects.index('person')

        # Filter images with invisible annotations
        if filter_invisible:  # FIXME add is_train? But then during detection images without boxes are fed
            vis_im_inds, annotations = zip(*[(i, ann) for i, ann in enumerate(annotations)
                                             if any([not inter['invis'] for inter in ann['interactions']])])
            num_old_images, num_new_images = len(image_ids), len(vis_im_inds)
            if num_new_images < num_old_images:
                print('Images have been discarded due to not having visible interactions. '
                      'Image index has changed (from %d images to %d).' % (num_old_images, num_new_images))
                image_ids = [image_ids[i] for i in vis_im_inds]
        assert len(annotations) == len(image_ids)

        # Compute COCO mapping
        coco_obj_to_idx = {v.replace(' ', '_'): k for k, v in COCO_CLASSES.items()}
        assert set(coco_obj_to_idx.keys()) - {'__background__'} == set(self._hicodet.objects)
        self._coco_to_hico_mapping = [coco_obj_to_idx[obj] for obj in self._objects]

        # Extract the data from Hico-DET annotations
        self.image_ids = image_ids
        self._im_boxes, self._im_box_classes, self._im_inters, self._im_without_visible_interactions, self._im_filenames = \
            self.compute_gt_data(annotations)
        assert not (filter_invisible and len(self._im_without_visible_interactions) > 0)
        assert len(self._im_boxes) == len(self._im_box_classes) == len(self._im_inters) == \
               len(annotations) == len(self.image_ids) == len(self._im_filenames)

        ################# Data augmentation pipeline
        pass  # You could add a data augmentation pipeline here, but we don't.

        ################# In case of precomputed features
        if self.is_train and cfg.program.load_precomputed_feats:
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
                pc_im_idx = np.flatnonzero(self.pc_image_ids == im_id).tolist()  # type: List
                assert len(pc_im_idx) == 1, pc_im_idx
                assert im_id not in self.im_id_to_pc_im_idx
                self.im_id_to_pc_im_idx[im_id] = pc_im_idx[0]
        else:
            self.pc_feats_file = None

    @property
    def objects(self):
        return self._objects

    @property
    def predicates(self):
        return self._predicates

    @property
    def person_class(self):
        return self._person_class_index

    @property
    def num_object_classes(self):
        return len(self.objects)

    @property
    def num_predicates(self):
        return len(self.predicates)

    @property
    def num_images(self):
        return len(self.image_ids)

    @property
    def img_dir(self):
        return self._hicodet.get_img_dir(self.split)

    @property
    def is_train(self):
        return self.split == Splits.TRAIN

    @property
    def coco_to_hico_mapping(self):
        return self._coco_to_hico_mapping

    @property
    def hois(self):
        # Each is (human, interaction, object)
        return np.concatenate(self._im_inters, axis=0)

    def compute_gt_data(self, annotations):
        predicate_index = {p: i for i, p in enumerate(self.predicates)}
        object_index = {o: i for i, o in enumerate(self.objects)}
        im_without_visible_interactions = []
        boxes, box_classes, interactions = [], [], []
        im_filenames = []
        for i, img_ann in enumerate(annotations):
            im_filenames.append(img_ann['file'])
            im_hum_boxes, im_obj_boxes, im_obj_box_classes, im_interactions = [], [], [], []
            for interaction in img_ann['interactions']:
                if not interaction['invis']:
                    curr_num_hum_boxes = int(sum([b.shape[0] for b in im_hum_boxes]))
                    curr_num_obj_boxes = int(sum([b.shape[0] for b in im_obj_boxes]))

                    # Interaction
                    pred_class = predicate_index[self._hicodet.interactions[interaction['id']]['pred']]
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
                    obj_class = object_index[self._hicodet.interactions[interaction['id']]['obj']]
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
                box_classes.append(np.concatenate([np.full(num_hum_boxes, fill_value=self.person_class, dtype=np.int), im_obj_box_classes]))
                interactions.append(im_interactions)
            else:
                boxes.append([])
                box_classes.append([])
                interactions.append([])
                im_without_visible_interactions.append(i)
        return boxes, box_classes, interactions, im_without_visible_interactions, im_filenames

    def get_loader(self, batch_size, num_workers=0, num_gpus=1, shuffle=None, drop_last=True, **kwargs):
        if shuffle is None:
            shuffle = True if self.is_train else False
        batch_size = batch_size * num_gpus
        if not self.is_train and batch_size > 1:
            print('Only single-image batches are supported during evaluation. Batch size changed from %d to 1.' % batch_size)
            batch_size = 1
        data_loader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda x: Minibatch.collate(x),
            drop_last=drop_last,
            # pin_memory=True,  # disable this in case of freezes
            **kwargs,
        )
        return data_loader

    def get_entry(self, idx, read_img=True):
        # Read the image
        img_fn = self._im_filenames[idx]
        img_id = self.image_ids[idx]
        gt_boxes = self._im_boxes[idx].astype(np.float, copy=False)

        if read_img:
            raw_image = cv2.imread(os.path.join(self.img_dir, img_fn))
            img_h, img_w = raw_image.shape[:2]
            flipped = self.is_train and np.random.random() < self.flipping_prob  # Optionally flip the image if we're doing training
            if flipped:
                raw_image = raw_image[:, ::-1, :]  # NOTE: change this to [:, :, ::-1] if the image is read through PIL
                gt_boxes[:, [0, 2]] = img_w - gt_boxes[:, [2, 0]]
            image, img_scale_factor = preprocess_img(raw_image)
            img_size = (img_h, img_w)
        else:
            image = img_size = img_scale_factor = flipped = None

        entry = Example(idx_in_split=idx, img_id=img_id, img_fn=img_fn,
                        gt_boxes=gt_boxes, gt_obj_classes=self._im_box_classes[idx].copy(), gt_hois=self._im_inters[idx].copy(),
                        image=image, img_size=img_size, img_scale_factor=img_scale_factor, flipped=flipped)

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
        return entry

    def __getitem__(self, idx):
        Timer.get('Epoch', 'GetBatch').tic()
        entry = self.get_entry(idx)
        Timer.get('Epoch', 'GetBatch').toc()
        return entry

    def __len__(self):
        return self.num_images


def main():
    cfg.parse_args()
    cfg.print()

    hd = HicoDetInstance(Splits.TRAIN, flipping_prob=cfg.data.flip_prob)


if __name__ == '__main__':
    main()
