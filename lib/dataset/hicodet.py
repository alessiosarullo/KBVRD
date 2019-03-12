import os
from typing import Dict
from typing import List

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from config import cfg
from lib.dataset.hicodet_driver import HicoDet as HicoDetDriver
from lib.dataset.utils import Splits, preprocess_img, Example, Minibatch
from lib.detection.wrappers import COCO_CLASSES
from lib.stats.utils import Timer


class HicoDetInstanceSplit(Dataset):
    _splits = {}  # type: Dict[Splits, HicoDetInstanceSplit]
    _hicodet_driver = None

    def __init__(self, split, hicodet_driver, annotations, image_ids, object_inds, predicate_inds, flipping_prob=0):
        """
        """
        # TODO docs, mention split in print so as not to have confusing messages
        assert split in Splits

        self.split = split
        self.image_ids = image_ids
        self.flipping_prob = flipping_prob

        self._annotations = annotations
        self._hicodet = hicodet_driver

        object_inds = sorted(object_inds)
        predicate_inds = sorted(predicate_inds)
        self.objects = [hicodet_driver.objects[i] for i in object_inds]
        self.predicates = [hicodet_driver.predicates[i] for i in predicate_inds]
        print('Flipping is %s.' % (('enabled with probability %.2f' % flipping_prob) if flipping_prob > 0 else 'disabled'))

        ################# Initialize
        self.human_class = self.objects.index('person')

        # Compute mappings to and from COCO
        coco_obj_to_idx = {c.replace(' ', '_'): i for i, c in COCO_CLASSES.items()}
        assert set(coco_obj_to_idx.keys()) - {'__background__'} == set(hicodet_driver.objects)
        self.hico_to_coco_mapping = np.array([coco_obj_to_idx[obj] for obj in self.objects], dtype=np.int)

        # Extract the data from Hico-DET annotations
        self._im_boxes, self._im_box_classes, self._im_inters, self._im_without_visible_interactions, self._im_filenames = \
            self.compute_gt_data(annotations)
        assert len(self._im_without_visible_interactions) == 0
        assert len(self._im_boxes) == len(self._im_box_classes) == len(self._im_inters) == \
               len(self._annotations) == len(self.image_ids) == len(self._im_filenames)

        ################# Data augmentation pipeline
        pass  # You could add a data augmentation pipeline here.

        ################# In case of precomputed features
        if cfg.program.load_precomputed_feats:
            assert self.flipping_prob == 0  # TODO? extract features for flipped image
            precomputed_feats_fn = cfg.program.precomputed_feats_file_format % (cfg.model.rcnn_arch,
                                                                                (Splits.TEST if self.split == Splits.TEST else Splits.TRAIN).value)
            print('Loading precomputed feats for %s split from %s.' % (self.split.value, precomputed_feats_fn))
            self.pc_feats_file = h5py.File(precomputed_feats_fn, 'r')
            assert self.pc_feats_file['box_feats'].shape[1] == self.pc_feats_file['union_boxes_feats'].shape[1] == 2048  # FIXME magic constant

            self.pc_box_im_inds = self.pc_feats_file['boxes_ext'][:, 0].astype(np.int)
            self.pc_hoi_infos = self.pc_feats_file['hoi_infos'][:].astype(np.int)
            self.pc_hoi_im_inds = self.pc_hoi_infos[:, 0]
            self.pc_image_infos = self.pc_feats_file['img_infos'][:]
            try:
                self.pc_box_labels = self.pc_feats_file['box_labels'][:]
            except KeyError:
                self.pc_box_labels = None

            # Map image IDs to indices over the precomputed image IDs
            self.pc_image_ids = self.pc_feats_file['image_ids'][:]
            assert len(set(self.image_ids) - set(self.pc_image_ids.tolist())) == 0
            assert len(self.pc_image_ids) == len(set(self.pc_image_ids))
            self.im_id_to_pc_im_idx = {}
            for im_id in self.image_ids:
                pc_im_idx = np.flatnonzero(self.pc_image_ids == im_id).tolist()  # type: List
                assert len(pc_im_idx) == 1, pc_im_idx
                assert im_id not in self.im_id_to_pc_im_idx
                self.im_id_to_pc_im_idx[im_id] = pc_im_idx[0]

            self.obj_class_inds = np.array(object_inds, dtype=np.int)
            self.hoi_class_inds = np.array(predicate_inds, dtype=np.int)
            self.box_ext_class_inds = np.concatenate([np.arange(5), 5 + self.obj_class_inds])
        else:
            self.pc_feats_file = None

    @classmethod
    def get_split(cls, split: Splits, im_inds=None, pred_inds=None, obj_inds=None, **kwargs):
        if split not in cls._splits:
            if split == Splits.VAL:
                assert Splits.TRAIN not in cls._splits or cfg.data.val_ratio == 0, 'Training split must be instantiated before validation split.'

            if cls._hicodet_driver is None:
                cls._hicodet_driver = HicoDetDriver()

            # Load inds from configs first. Note that these might still be None after this step, which means all possible indices will be used.
            im_inds = im_inds or cfg.data.im_inds
            obj_inds = obj_inds or cfg.data.obj_inds
            pred_inds = pred_inds or cfg.data.pred_inds

            annotations, image_ids, object_inds, predicate_inds = compute_annotations(split, cls._hicodet_driver, im_inds, obj_inds, pred_inds)
            assert len(annotations) == len(image_ids)

            # Split train/val if needed
            if cfg.data.val_ratio > 0 and split == Splits.TRAIN:
                num_val_imgs = int(len(annotations) * cfg.data.val_ratio)
                cls._splits[Splits.TRAIN] = cls(split=Splits.TRAIN, hicodet_driver=cls._hicodet_driver,
                                                annotations=annotations[:-num_val_imgs], image_ids=image_ids[:-num_val_imgs],
                                                object_inds=object_inds, predicate_inds=predicate_inds,
                                                **kwargs)
                cls._splits[Splits.VAL] = cls(split=Splits.VAL, hicodet_driver=cls._hicodet_driver,
                                              annotations=annotations[-num_val_imgs:], image_ids=image_ids[-num_val_imgs:],
                                              object_inds=object_inds, predicate_inds=predicate_inds,
                                              **kwargs)
            else:
                cls._splits[split] = cls(split=split, annotations=annotations, hicodet_driver=cls._hicodet_driver,
                                         image_ids=image_ids, object_inds=object_inds, predicate_inds=predicate_inds,
                                         **kwargs)

        return cls._splits[split]

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
        split = Splits.TEST if self.split == Splits.TEST else Splits.TRAIN
        return self._hicodet.get_img_dir(split)

    @property
    def hois(self):
        # Each is (human, interaction, object)
        return np.concatenate(self._im_inters, axis=0)

    @property
    def obj_labels(self):
        return np.concatenate(self._im_box_classes, axis=0)

    @property
    def has_precomputed(self):
        return self.pc_feats_file is not None

    @property
    def precomputed_visual_feat_dim(self):
        if not self.has_precomputed:
            raise AttributeError('No precomputed visual features are present.')
        return self.pc_feats_file['box_feats'].shape[1]

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
                box_classes.append(np.concatenate([np.full(num_hum_boxes, fill_value=self.human_class, dtype=np.int), im_obj_box_classes]))
                interactions.append(im_interactions)
            else:
                boxes.append([])
                box_classes.append([])
                interactions.append([])
                im_without_visible_interactions.append(i)
        return boxes, box_classes, interactions, im_without_visible_interactions, im_filenames

    def get_loader(self, batch_size, num_workers=0, num_gpus=1, shuffle=None, drop_last=True, **kwargs):
        if shuffle is None:
            shuffle = True if self.split == Splits.TRAIN else False
        batch_size = batch_size * num_gpus
        if self.split == Splits.TEST and batch_size > 1:
            print('Only single-image batches are supported during prediction. Batch size changed from %d to 1.' % batch_size)
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

        entry = Example(idx_in_split=idx, img_id=img_id, img_fn=img_fn, precomputed=self.has_precomputed)
        if not self.has_precomputed:
            gt_boxes = self._im_boxes[idx].astype(np.float, copy=False)

            if read_img:
                raw_image = cv2.imread(os.path.join(self.img_dir, img_fn))
                img_h, img_w = raw_image.shape[:2]
                flipped = self.split == Splits.TRAIN and np.random.random() < self.flipping_prob  # Optionally flip the image if we're doing training
                if flipped:
                    raw_image = raw_image[:, ::-1, :]  # NOTE: change this to [:, :, ::-1] if the image is read through PIL
                    gt_boxes[:, [0, 2]] = img_w - gt_boxes[:, [2, 0]]
                image, img_scale_factor = preprocess_img(raw_image)
                img_size = (img_h, img_w)

                entry.image = image
                entry.img_size = img_size
                entry.scale = img_scale_factor
                entry.flipped = flipped

            entry.gt_boxes = gt_boxes
            entry.gt_obj_classes = self._im_box_classes[idx].copy()
            entry.gt_hois = self._im_inters[idx].copy()
        else:
            pc_im_idx = self.im_id_to_pc_im_idx[img_id]
            assert self.pc_image_ids[pc_im_idx] == img_id, (self.pc_image_ids[pc_im_idx], img_id)

            # Image data
            img_infos = self.pc_image_infos[pc_im_idx]
            assert img_infos.shape == (3,)
            entry.img_size = img_infos[:2]
            entry.scale = img_infos[2]

            # Object data
            img_box_inds = np.flatnonzero(self.pc_box_im_inds == pc_im_idx)
            if img_box_inds.size > 0:
                start, end = img_box_inds[0], img_box_inds[-1] + 1
                assert np.all(img_box_inds == np.arange(start, end))  # slicing is much more efficient with H5 files

                precomp_boxes_ext = self.pc_feats_file['boxes_ext'][start:end, :]
                precomp_boxes_ext = precomp_boxes_ext[:, self.box_ext_class_inds]
                precomp_box_feats = self.pc_feats_file['box_feats'][start:end, :]
                precomp_masks = self.pc_feats_file['masks'][start:end, :, :]
                if self.pc_box_labels is not None:
                    # TODO mapping of obj inds for rels
                    precomp_box_labels = self.pc_box_labels[start:end]
                    num_boxes = precomp_box_labels.shape[0]
                    precomp_box_labels_one_hot = np.zeros([num_boxes, len(self._hicodet_driver.objects)], dtype=precomp_box_labels.dtype)
                    precomp_box_labels_one_hot[np.arange(num_boxes), precomp_box_labels] = 1
                    precomp_box_labels_one_hot = precomp_box_labels_one_hot[:, self.obj_class_inds]
                    feasible_box_labels_inds = np.any(precomp_box_labels_one_hot, axis=1)

                    box_inds = np.full_like(precomp_box_labels, fill_value=-1)
                    box_inds[feasible_box_labels_inds] = np.arange(np.sum(feasible_box_labels_inds))
                    precomp_boxes_ext = precomp_boxes_ext[feasible_box_labels_inds]
                    precomp_box_feats = precomp_box_feats[feasible_box_labels_inds]
                    precomp_masks = precomp_masks[feasible_box_labels_inds]
                    precomp_box_labels_one_hot = precomp_box_labels_one_hot[feasible_box_labels_inds]
                    assert np.all(np.sum(precomp_box_labels_one_hot, axis=1) == 1), precomp_box_labels_one_hot
                    x = np.where(precomp_box_labels_one_hot)
                    assert np.all(x[0] == np.arange(np.sum(feasible_box_labels_inds)))
                    precomp_box_labels = x[1].astype(precomp_box_labels_one_hot.dtype)
                else:
                    precomp_box_labels = None
                    box_inds = None

                # HOI data
                img_hoi_inds = np.flatnonzero(self.pc_hoi_im_inds == pc_im_idx)
                assert img_hoi_inds.size > 0, (idx, pc_im_idx, img_id, img_fn)

                start, end = img_hoi_inds[0], img_hoi_inds[-1] + 1
                assert np.all(img_hoi_inds == np.arange(start, end))  # slicing is much more efficient with H5 files
                precomp_hoi_infos = self.pc_hoi_infos[start:end, :]
                precomp_hoi_union_boxes = self.pc_feats_file['union_boxes'][start:end, :]
                precomp_hoi_union_feats = self.pc_feats_file['union_boxes_feats'][start:end, :]
                try:
                    precomp_hoi_labels = self.pc_feats_file['hoi_labels'][start:end, :]
                except KeyError:
                    precomp_hoi_labels = None

                if precomp_hoi_labels is not None:
                    assert precomp_box_labels is not None and box_inds is not None
                    precomp_hoi_labels = precomp_hoi_labels[:, self.hoi_class_inds]

                    # Remap HOIs box indices
                    precomp_hoi_infos[:, 1] = box_inds[precomp_hoi_infos[:, 1]]
                    precomp_hoi_infos[:, 2] = box_inds[precomp_hoi_infos[:, 2]]

                    # Filter out HOIs
                    feasible_hoi_labels_inds = np.any(precomp_hoi_labels, axis=1) & np.all(precomp_hoi_infos >= 0, axis=1)
                    precomp_hoi_infos = precomp_hoi_infos[feasible_hoi_labels_inds]
                    precomp_hoi_union_boxes = precomp_hoi_union_boxes[feasible_hoi_labels_inds]
                    precomp_hoi_union_feats = precomp_hoi_union_feats[feasible_hoi_labels_inds]
                    precomp_hoi_labels = precomp_hoi_labels[feasible_hoi_labels_inds, :]
                    assert np.all(np.sum(precomp_hoi_labels, axis=1) >= 1), precomp_hoi_labels

                    # Filter out boxes without interactions
                    hoi_box_inds = np.unique(precomp_hoi_infos[:, 1:])
                    # if np.any(hoi_box_inds != np.arange(hoi_box_inds.shape[0)):
                    #     print('Bingpot!')  # FIXME
                    precomp_boxes_ext = precomp_boxes_ext[hoi_box_inds]
                    precomp_box_feats = precomp_box_feats[hoi_box_inds]
                    precomp_masks = precomp_masks[hoi_box_inds]
                    precomp_box_labels = precomp_box_labels[hoi_box_inds]
                    box_inds = np.full(np.amax(hoi_box_inds) + 1, fill_value=-1)
                    box_inds[hoi_box_inds] = np.arange(hoi_box_inds.shape[0])
                    precomp_hoi_infos[:, 1] = box_inds[precomp_hoi_infos[:, 1]]
                    precomp_hoi_infos[:, 2] = box_inds[precomp_hoi_infos[:, 2]]
                    assert np.all(precomp_hoi_infos >= 0), precomp_hoi_infos

                assert np.all(precomp_hoi_infos >= 0)
                entry.precomp_hoi_labels = precomp_hoi_labels
                entry.precomp_hoi_infos = precomp_hoi_infos
                entry.precomp_hoi_union_boxes = precomp_hoi_union_boxes
                entry.precomp_hoi_union_feats = precomp_hoi_union_feats

                entry.precomp_boxes_ext = precomp_boxes_ext
                entry.precomp_box_feats = precomp_box_feats
                entry.precomp_masks = precomp_masks
                entry.precomp_box_labels = precomp_box_labels
            assert (entry.precomp_box_labels is None and entry.precomp_hoi_labels is None) or \
                   (entry.precomp_box_labels is not None and entry.precomp_hoi_labels is not None)
        return entry

    def __getitem__(self, idx):
        Timer.get('GetBatch').tic()
        entry = self.get_entry(idx)
        Timer.get('GetBatch').toc()
        return entry

    def __len__(self):
        return self.num_images


def compute_annotations(split, hicodet_driver, im_inds, obj_inds, pred_inds, filter_invisible=True):
    # Set annotations and image index
    annotations = hicodet_driver.split_data[split if split == Splits.TEST else Splits.TRAIN]['annotations']
    image_ids = im_inds or list(range(len(annotations)))
    if im_inds is not None:
        annotations = [annotations[i] for i in im_inds]

    # Filter out unwanted predicates/object classes
    if obj_inds is None and pred_inds is None:
        final_objects_inds = list(range(len(hicodet_driver.objects)))
        final_pred_inds = list(range(len(hicodet_driver.predicates)))
    else:
        obj_inds = set(obj_inds or range(len(hicodet_driver.objects)))
        pred_inds = set(pred_inds or range(len(hicodet_driver.predicates)))
        assert 0 in pred_inds
        new_im_inds, new_annotations = [], []
        pred_count = {}
        obj_count = {}
        for i, im_ann in enumerate(annotations):
            new_im_inters = []
            for inter in im_ann['interactions']:
                ann_obj = hicodet_driver.get_object_index(inter['id'])
                ann_pred = hicodet_driver.get_predicate_index(inter['id'])
                if ann_obj in obj_inds and ann_pred in pred_inds:
                    new_im_inters.append(inter)
                    pred_count[ann_pred] = pred_count.get(ann_pred, 0) + 1
                    obj_count[ann_obj] = obj_count.get(ann_obj, 0) + 1
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
        # Also, if both predicate and object indices are specified, some of them might not be present due to not having suitable predicate-object
        # pairs. These will be removed, as the model can't actually train on them due to the lack of examples.
        final_objects_inds = sorted(set(obj_count.keys()) | {hicodet_driver.human_class})
        final_pred_inds = sorted(set(pred_count.keys()))
        if pred_inds - set(pred_count.keys()):
            print('The following predicates have been discarded due to the lack of feasible objects: %s.' %
                  ', '.join(['%s (%d)' % (hicodet_driver.predicates[p], p) for p in (pred_inds - set(pred_count.keys()))]))
        if obj_inds - set(obj_count.keys()):
            print('The following objects have been discarded due to the lack of feasible predicates: %s.' %
                  ', '.join(['%s (%d)' % (hicodet_driver.objects[o], o) for o in (obj_inds - set(obj_count.keys()))]))

    # Filter images with invisible annotations
    if filter_invisible:
        vis_im_inds, annotations = zip(*[(i, ann) for i, ann in enumerate(annotations)
                                         if any([not inter['invis'] for inter in ann['interactions']])])
        num_old_images, num_new_images = len(image_ids), len(vis_im_inds)
        if num_new_images < num_old_images:
            print('Images have been discarded due to not having visible interactions. '
                  'Image index has changed (from %d images to %d).' % (num_old_images, num_new_images))
            image_ids = [image_ids[i] for i in vis_im_inds]
    assert len(annotations) == len(image_ids)

    return annotations, image_ids, final_objects_inds, final_pred_inds


def main():
    import sys
    sys.argv += ['--model', 'base', '--save_dir', 'fake']
    cfg.parse_args()

    hd = HicoDetInstanceSplit.get_split(split=Splits.TRAIN, flipping_prob=cfg.data.flip_prob)
    for im_i, hois in enumerate(hd._im_inters):
        # u_hois, counts = np.unique(hois[:, [0, 2]], return_counts=True)

        hh_inter = np.any(hois[:, 0] == hois[:, 2])
        counts = np.sum(hh_inter)

        if np.any(counts > 1):
            print(im_i, np.sum(counts > 1))


if __name__ == '__main__':
    main()
