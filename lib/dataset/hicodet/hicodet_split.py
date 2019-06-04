import os
from typing import Dict, List, Type, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import cfg
from lib.dataset.hicodet.hicodet import HicoDet, HicoDetImData
from lib.dataset.utils import Splits, preprocess_img, im_list_to_4d_tensor
from lib.detection.wrappers import COCO_CLASSES


class Example:
    def __init__(self, idx_in_split, img_id, filename, split):
        self.index = idx_in_split
        self.id = img_id
        self.filename = filename
        self.split = split

        self.img_size = None
        self.scale = None

        self.image = None
        self.gt_boxes = None
        self.gt_obj_classes = None
        self.gt_hois = None


class Minibatch:
    # TODO refactor in list[Example], then merged when called vectorize()
    def __init__(self):
        self.img_infos = []
        self.other_ex_data = []

        self.imgs = []
        self.gt_boxes = []
        self.gt_obj_classes = []
        self.gt_box_im_ids = []
        self.gt_hois = []
        self.gt_hoi_im_ids = []

    def append(self, ex: Example):
        self.img_infos += [np.array([*ex.img_size, ex.scale], dtype=np.float32)]

        self.other_ex_data += [{'index': ex.index,
                                'id': ex.id,
                                'fn': ex.filename,
                                'split': ex.split,
                                'im_size': ex.img_size,  # this won't be changed
                                'im_scale': ex.scale,
                                }]

        self.imgs += [ex.image]
        self.gt_boxes += [ex.gt_boxes * ex.scale]
        self.gt_obj_classes += [ex.gt_obj_classes]
        self.gt_hois += [ex.gt_hois]

        self.gt_box_im_ids += [np.full_like(ex.gt_obj_classes, fill_value=len(self.gt_box_im_ids))]
        self.gt_hoi_im_ids += [np.full(ex.gt_hois.shape[0], fill_value=len(self.gt_hoi_im_ids), dtype=np.int)]

    def vectorize(self, device):
        assert all([len(v) > 0 for k, v in self.__dict__.items() if k.startswith('gt_')])

        self.imgs = im_list_to_4d_tensor([torch.tensor(v, device=device) for v in self.imgs])  # 4D NCHW tensor

        img_infos = np.stack(self.img_infos, axis=0)
        img_infos[:, 0] = self.imgs.shape[2]
        img_infos[:, 1] = self.imgs.shape[3]
        self.img_infos = torch.tensor(img_infos, dtype=torch.float32, device=device)

        assert self.imgs.shape[0] == self.img_infos.shape[0]

        self.gt_box_im_ids = np.concatenate(self.gt_box_im_ids, axis=0)
        self.gt_boxes = np.concatenate(self.gt_boxes, axis=0)
        self.gt_obj_classes = np.concatenate(self.gt_obj_classes, axis=0)
        self.gt_hoi_im_ids = np.concatenate(self.gt_hoi_im_ids, axis=0)
        self.gt_hois = np.concatenate(self.gt_hois, axis=0)
        assert len(self.gt_boxes) == len(self.gt_obj_classes) == len(self.gt_box_im_ids)
        assert len(self.gt_hois) == len(self.gt_hoi_im_ids)

    @classmethod
    def collate(cls, examples):
        minibatch = cls()
        for ex in examples:
            minibatch.append(ex)
        minibatch.vectorize(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return minibatch


class HicoDetSplit(Dataset):
    def __init__(self, split, hicodet: HicoDet, data: List[HicoDetImData], image_ids, object_inds, predicate_inds):
        assert split in Splits

        self.split = split
        self.hicodet = hicodet  # type: HicoDet
        self.image_ids = image_ids
        self._data = data

        object_inds = sorted(object_inds)
        self.objects = [hicodet.objects[i] for i in object_inds]
        self.obj_class_inds = np.array(object_inds, dtype=np.int)
        self.object_index = {obj: i for i, obj in enumerate(self.objects)}

        predicate_inds = sorted(predicate_inds)
        self.predicates = [hicodet.predicates[i] for i in predicate_inds]
        self.action_class_inds = np.array(predicate_inds, dtype=np.int)
        self.predicate_index = {pred: i for i, pred in enumerate(self.predicates)}

        interactions = np.array([[self.predicate_index.get(self.hicodet.predicates[p], -1), self.object_index.get(self.hicodet.objects[o], -1)]
                                 for p, o in self.hicodet.interactions])
        self.interactions = interactions[np.all(interactions >= 0, axis=1), :]

        # Compute mappings to and from COCO
        coco_obj_to_idx = {('hair dryer' if c == 'hair drier' else c).replace(' ', '_'): i for i, c in COCO_CLASSES.items()}
        assert set(coco_obj_to_idx.keys()) - {'__background__'} == set(hicodet.objects)
        self.hico_to_coco_mapping = np.array([coco_obj_to_idx[obj] for obj in self.objects], dtype=np.int)

        # Compute HOI triplets. Each is [human, interaction, object].
        hoi_triplets = []
        for im_data in self._data:
            box_classes, inters = im_data.box_classes, im_data.interactions
            im_hois = np.stack([box_classes[inters[:, 0]], inters[:, 1], box_classes[inters[:, 2]]], axis=1)
            assert np.all(im_hois[:, 0] == self.human_class)
            hoi_triplets.append(im_hois)
        self.hoi_triplets = np.concatenate(hoi_triplets, axis=0)

        self.obj_labels = np.concatenate([im_data.box_classes for im_data in self._data])

    @property
    def human_class(self) -> int:
        return self.hicodet.human_class

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
        split = Splits.TEST if self.split == Splits.TEST else Splits.TRAIN  # val -> train
        return self.hicodet.get_img_dir(split)

    def get_preds_for_embs(self, keep_prepositions=False):
        preds = []
        for p in self.predicates:
            if p != self.hicodet.null_interaction:
                p = p.replace('_', ' ')
            if not keep_prepositions:
                p = p.split(' ')[0]
            preds.append(p)
        return preds

    def get_img_entry(self, idx, read_img=True) -> Example:
        im_data = self._data[idx]

        entry = Example(idx_in_split=idx, img_id=self.image_ids[idx], filename=im_data.filename, split=self.split)
        if read_img:
            raw_image = cv2.imread(os.path.join(self.img_dir, im_data.filename))
            img_h, img_w = raw_image.shape[:2]
            image, img_scale_factor = preprocess_img(raw_image)
            img_size = (img_h, img_w)

            entry.image = image
            entry.img_size = img_size
            entry.scale = img_scale_factor

        entry.gt_boxes = im_data.boxes.astype(np.float, copy=False)
        entry.gt_obj_classes = im_data.box_classes.copy()
        entry.gt_hois = im_data.interactions.copy()
        return entry

    def __getitem__(self, idx):
        return self.get_img_entry(idx)

    def __len__(self):
        return self.num_images


class HicoDetSplitBuilder:
    splits = {}  # type: Dict[Type[HicoDetSplit], Dict[Splits, HicoDetSplit]]
    hicodet = None

    def __init__(self):
        raise NotImplementedError('Use class methods only.')

    @classmethod
    def get_split(cls, split_class: Type[HicoDetSplit], split: Splits, pred_inds=None, obj_inds=None):
        class_splits = cls.splits.setdefault(split_class, {})
        if split not in class_splits:
            if split == Splits.VAL:
                assert Splits.TRAIN not in class_splits or cfg.data.val_ratio == 0, 'Training split must be instantiated before validation split.'

            if cls.hicodet is None:
                cls.hicodet = HicoDet()

            # Load inds from configs first. Note that these might still be None after this step, which means all possible indices will be used.
            obj_inds = obj_inds or cfg.data.obj_inds
            pred_inds = pred_inds or cfg.data.pred_inds

            split_data, image_ids, object_inds, predicate_inds = filter_data(split, cls.hicodet, obj_inds, pred_inds,
                                                                             filter_empty_imgs=split == Splits.TRAIN)
            assert len(split_data) == len(image_ids)

            # Split train/val if needed
            if cfg.data.val_ratio > 0 and split == Splits.TRAIN:
                num_val_imgs = int(len(split_data) * cfg.data.val_ratio)
                class_splits[Splits.TRAIN] = split_class(split=Splits.TRAIN, hicodet=cls.hicodet,
                                                         data=split_data[:-num_val_imgs], image_ids=image_ids[:-num_val_imgs],
                                                         object_inds=object_inds, predicate_inds=predicate_inds)
                class_splits[Splits.VAL] = split_class(split=Splits.VAL, hicodet=cls.hicodet,
                                                       data=split_data[-num_val_imgs:], image_ids=image_ids[-num_val_imgs:],
                                                       object_inds=object_inds, predicate_inds=predicate_inds)
            else:
                class_splits[split] = split_class(split=split, hicodet=cls.hicodet, data=split_data, image_ids=image_ids,
                                                  object_inds=object_inds, predicate_inds=predicate_inds)

        return class_splits[split]

    @classmethod
    def get_splits(cls, hdsplit_class: Type[HicoDetSplit], splits: Union[List[Splits], Splits], pred_inds=None, obj_inds=None):
        if not isinstance(splits, List):
            splits = [splits]
        if cls.hicodet is None:
            cls.hicodet = HicoDet()
        if Splits.VAL in splits:
            assert Splits.TRAIN in splits, 'Validation split requires train.'
            assert cfg.data.val_ratio > 0
            val = True
        else:
            val = False

        class_splits = {}
        for split in [s for s in splits if s != Splits.VAL]:
            # Load inds from configs first. Note that these might still be None after this step, which means all possible indices will be used.
            obj_inds = obj_inds or cfg.data.obj_inds
            pred_inds = pred_inds or cfg.data.pred_inds

            split_data, image_ids, object_inds, predicate_inds = filter_data(split, cls.hicodet, obj_inds, pred_inds,
                                                                             filter_empty_imgs=split == Splits.TRAIN)
            assert len(split_data) == len(image_ids)

            # Split train/val if needed
            if val:
                num_val_imgs = int(len(split_data) * cfg.data.val_ratio)
                class_splits[Splits.TRAIN] = hdsplit_class(split=Splits.TRAIN, hicodet=cls.hicodet,
                                                           data=split_data[:-num_val_imgs], image_ids=image_ids[:-num_val_imgs],
                                                           object_inds=object_inds, predicate_inds=predicate_inds)
                class_splits[Splits.VAL] = hdsplit_class(split=Splits.VAL, hicodet=cls.hicodet,
                                                         data=split_data[-num_val_imgs:], image_ids=image_ids[-num_val_imgs:],
                                                         object_inds=object_inds, predicate_inds=predicate_inds)
            else:
                class_splits[split] = hdsplit_class(split=split, hicodet=cls.hicodet, data=split_data, image_ids=image_ids,
                                                    object_inds=object_inds, predicate_inds=predicate_inds)

        ret = [class_splits[s] for s in splits]
        if len(ret) == 1:
            ret = ret[0]
        return ret


def remap_box_pairs(box_pairs, box_mask):
    box_inds = np.full(box_mask.shape[0], fill_value=-1, dtype=np.int)
    box_inds[box_mask] = np.arange(np.sum(box_mask))
    box_pairs[:, 0] = box_inds[box_pairs[:, 0]]
    box_pairs[:, 1] = box_inds[box_pairs[:, 1]]
    return box_pairs


def filter_data(split, hicodet: HicoDet, obj_inds, pred_inds, filter_empty_imgs):
    split_data = hicodet.split_data[split]  # type: List[HicoDetImData]
    image_ids = list(range(len(split_data)))

    # Filter out unwanted predicates/object classes
    if obj_inds is None and pred_inds is None:
        final_objects_inds = list(range(len(hicodet.objects)))
        final_pred_inds = list(range(len(hicodet.predicates)))
    else:
        obj_inds = set(obj_inds or range(len(hicodet.objects)))
        pred_inds = set(pred_inds or range(len(hicodet.predicates)))
        assert 0 in pred_inds
        new_im_inds, new_split_data = [], []
        pred_count = {}
        obj_count = {}
        for i, im_data in enumerate(split_data):
            boxes, box_classes, interactions = im_data.boxes, im_data.box_classes, im_data.interactions

            box_mask = np.array([c in obj_inds for i, c in enumerate(box_classes)], dtype=bool)
            boxes = boxes[box_mask, :]
            box_classes = box_classes[box_mask]
            interactions[:, [0, 2]] = remap_box_pairs(interactions[:, [0, 2]], box_mask)
            interactions = interactions[np.all(interactions >= 0, axis=1), :]

            if interactions.size > 0:
                for pred in interactions[:, 1]:
                    pred_count[pred] = pred_count.get(pred, 0) + 1
                for obj in box_classes:
                    obj_count[obj] = obj_count.get(obj, 0) + 1

                new_im_inds.append(i)
                new_split_data.append(HicoDetImData(filename=im_data.filename, boxes=boxes, box_classes=box_classes, interactions=interactions))

        num_inters = sum([im_data.interactions.shape[0] for im_data in split_data])
        diff_num_inters = num_inters - sum([im_data.interactions.shape[0] for im_data in new_split_data])
        if diff_num_inters > 0:
            print('%d/%d interaction%s been filtered out.' % (diff_num_inters, num_inters, ' has' if diff_num_inters == 1 else 's have'))
        if len(new_im_inds) < len(image_ids):
            print('Images have been discarded due to not having feasible predicates or objects. '
                  'Image index has changed (from %d images to %d).' % (len(image_ids), len(new_im_inds)))
        split_data = new_split_data
        image_ids = [image_ids[i] for i in new_im_inds]

        # Now we can add the person class: there won't be interactions with persons as an object if not in the initial indices, but the dataset
        # includes the class anyway because the model must always be able to predict it.
        # Also, if both predicate and object indices are specified, some of them might not be present due to not having suitable predicate-object
        # pairs. These will be removed, as the model can't actually train on them due to the lack of examples.
        final_objects_inds = sorted(set(obj_count.keys()) | {hicodet.human_class})
        final_pred_inds = sorted(set(pred_count.keys()))
        if pred_inds - set(pred_count.keys()):
            print('The following predicates have been discarded due to the lack of feasible objects: %s.' %
                  ', '.join(['%s (%d)' % (hicodet.predicates[p], p) for p in (pred_inds - set(pred_count.keys()))]))
        if obj_inds - set(obj_count.keys()):
            print('The following objects have been discarded due to the lack of feasible predicates: %s.' %
                  ', '.join(['%s (%d)' % (hicodet.objects[o], o) for o in (obj_inds - set(obj_count.keys()))]))

    # Filter images with only invisible or null interactions
    if filter_empty_imgs:
        im_with_interactions, new_split_data = [], []
        for i, im_data in enumerate(split_data):
            if np.any(im_data.interactions[:, 1] != hicodet.predicate_index[hicodet.null_interaction]):
                im_with_interactions.append(i)
                new_split_data.append(im_data)
        split_data = new_split_data
        num_old_images, num_new_images = len(image_ids), len(im_with_interactions)
        if num_new_images < num_old_images:
            print('Images have been discarded due to not having visible interactions or only having background ones. '
                  'Image index has changed (from %d images to %d).' % (num_old_images, num_new_images))
            image_ids = [image_ids[i] for i in im_with_interactions]
    assert len(split_data) == len(image_ids)

    return split_data, image_ids, final_objects_inds, final_pred_inds
