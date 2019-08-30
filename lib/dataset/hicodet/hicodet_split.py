import os
from typing import Dict, List, Type

import cv2
import numpy as np
import torch

from config import cfg
from lib.dataset.hicodet.hicodet import HicoDet, HicoDetImData
from lib.dataset.hoi_dataset_split import AbstractHoiDatasetSplit
from lib.dataset.utils import Splits, preprocess_img, im_list_to_4d_tensor, get_hico_to_coco_mapping


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
                                'im_size': np.array(ex.img_size),  # this won't be changed
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
        self.img_infos = img_infos.astype(np.float32)

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


class HicoDetSplit(AbstractHoiDatasetSplit):
    def __init__(self, split, hicodet: HicoDet, data: List[HicoDetImData], image_ids, object_inds, predicate_inds, inter_inds):
        assert split in Splits

        self.split = split
        self.full_dataset = hicodet  # type: HicoDet
        self.image_ids = image_ids
        self._data = data

        object_inds = sorted(object_inds)
        self.objects = [hicodet.objects[i] for i in object_inds]
        self.active_object_classes = np.array(object_inds, dtype=np.int)
        reduced_object_index = {obj: i for i, obj in enumerate(self.objects)}
        if len(object_inds) < self.full_dataset.num_objects:
            print(f'{split.value.capitalize()} objects:', object_inds)

        predicate_inds = sorted(predicate_inds)
        self.predicates = [hicodet.predicates[i] for i in predicate_inds]
        self.active_actions = np.array(predicate_inds, dtype=np.int)
        reduced_predicate_index = {pred: i for i, pred in enumerate(self.predicates)}
        if len(predicate_inds) < self.full_dataset.num_actions:
            print(f'{split.value.capitalize()} predicates:', predicate_inds)

        if inter_inds is not None:
            assert len(object_inds) == self.full_dataset.num_objects and len(predicate_inds) == self.full_dataset.num_actions
            inter_inds = sorted(inter_inds)
            self.active_interactions = np.array(inter_inds, dtype=np.int)
            self.interactions = self.full_dataset.interactions[self.active_interactions, :]  # original predicate and object inds
            self.reduced_interactions = self.interactions
            print(f'{split.value.capitalize()} interactions:', inter_inds)
        else:
            reduced_interactions = np.array([[reduced_predicate_index.get(self.full_dataset.predicates[p], -1),
                                              reduced_object_index.get(self.full_dataset.objects[o], -1)]
                                             for p, o in self.full_dataset.interactions])
            self.reduced_interactions = reduced_interactions[np.all(reduced_interactions >= 0, axis=1), :]  # reduced predicate and object inds
            active_interactions = set(np.unique(self.full_dataset.op_pair_to_interaction[:, self.active_actions]).tolist()) - {-1}
            self.active_interactions = np.array(sorted(active_interactions), dtype=np.int)
            self.interactions = self.full_dataset.interactions[self.active_interactions, :]  # original predicate and object inds

            # Checks
            interactions = np.array([[p if self.full_dataset.predicates[p] in reduced_predicate_index else -1,
                                      o if self.full_dataset.objects[o] in reduced_object_index else -1]
                                     for p, o in self.full_dataset.interactions])
            assert np.all(self.interactions == interactions[np.all(interactions >= 0, axis=1), :])
            assert np.all([reduced_predicate_index[self.full_dataset.predicates[p]] == self.reduced_interactions[i, 0] and
                           reduced_object_index[self.full_dataset.objects[o]] == self.reduced_interactions[i, 1]
                           for i, (p, o) in enumerate(self.interactions)])

        # Compute mappings to COCO
        self.hico_to_coco_mapping = get_hico_to_coco_mapping(hico_objects=hicodet.objects, split_objects=self.objects)

        # Compute HOI triplets. Each is [human, action, object].
        hoi_triplets = []
        for im_data in self._data:
            box_classes, inters = im_data.box_classes, im_data.hois
            im_hois = np.stack([box_classes[inters[:, 0]], inters[:, 1], box_classes[inters[:, 2]]], axis=1)
            assert np.all(im_hois[:, 0] == self.human_class)
            hoi_triplets.append(im_hois)
        self.hoi_triplets = np.concatenate(hoi_triplets, axis=0)

        self.obj_labels = np.concatenate([im_data.box_classes for im_data in self._data])

    @property
    def human_class(self) -> int:
        return self.full_dataset.human_class

    @property
    def num_objects(self):
        return len(self.objects)

    @property
    def num_actions(self):
        return len(self.predicates)

    @property
    def num_interactions(self):
        return self.interactions.shape[0]

    @property
    def num_images(self):
        return len(self.image_ids)

    @property
    def img_dir(self):
        split = Splits.TEST if self.split == Splits.TEST else Splits.TRAIN  # val -> train
        return self.full_dataset.get_img_dir(split)

    def get_preds_for_embs(self, keep_prepositions=False):
        preds = []
        for p in self.predicates:
            if p != self.full_dataset.null_interaction:
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
            img_size = [img_h, img_w]

            entry.image = image
            entry.img_size = img_size
            entry.scale = img_scale_factor

        entry.gt_boxes = im_data.boxes.astype(np.float, copy=False)
        entry.gt_obj_classes = im_data.box_classes.copy()
        entry.gt_hois = im_data.hois.copy()
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
    def get_split(cls, split_class: Type[HicoDetSplit], split: Splits, pred_inds=None, obj_inds=None, inter_inds=None):
        class_splits = cls.splits.setdefault(split_class, {})
        if split not in class_splits:
            if split == Splits.VAL:
                assert Splits.TRAIN not in class_splits or cfg.val_ratio == 0, 'Training split must be instantiated before validation split.'

            if cls.hicodet is None:
                cls.hicodet = HicoDet()

            split_data, image_ids, object_inds, predicate_inds = filter_data(split, cls.hicodet, obj_inds, pred_inds, inter_inds,
                                                                             filter_empty_imgs=split == Splits.TRAIN,
                                                                             filter_null_imgs=(split == Splits.TRAIN and cfg.filter_bg_only))
            assert len(split_data) == len(image_ids)

            # Split train/val if needed
            if cfg.val_ratio > 0 and split == Splits.TRAIN:
                num_val_imgs = int(len(split_data) * cfg.val_ratio)
                class_splits[Splits.TRAIN] = split_class(split=Splits.TRAIN, hicodet=cls.hicodet,
                                                         data=split_data[:-num_val_imgs], image_ids=image_ids[:-num_val_imgs],
                                                         object_inds=object_inds, predicate_inds=predicate_inds, inter_inds=inter_inds)
                class_splits[Splits.VAL] = split_class(split=Splits.VAL, hicodet=cls.hicodet,
                                                       data=split_data[-num_val_imgs:], image_ids=image_ids[-num_val_imgs:],
                                                       object_inds=object_inds, predicate_inds=predicate_inds, inter_inds=inter_inds)
            else:
                class_splits[split] = split_class(split=split, hicodet=cls.hicodet, data=split_data, image_ids=image_ids,
                                                  object_inds=object_inds, predicate_inds=predicate_inds, inter_inds=inter_inds)

        return class_splits[split]

    # @classmethod
    # def get_splits(cls, hdsplit_class: Type[HicoDetSplit], splits: Union[List[Splits], Splits], pred_inds=None, obj_inds=None):
    #     if not isinstance(splits, List):
    #         splits = [splits]
    #     if cls.hicodet is None:
    #         cls.hicodet = HicoDet()
    #     if Splits.VAL in splits:
    #         assert Splits.TRAIN in splits, 'Validation split requires train.'
    #         assert cfg.val_ratio > 0
    #         val = True
    #     else:
    #         val = False
    #
    #     class_splits = {}
    #     for split in [s for s in splits if s != Splits.VAL]:
    #         # Load inds from configs first. Note that these might still be None after this step, which means all possible indices will be used.
    #         obj_inds = obj_inds or cfg.obj_inds
    #         pred_inds = pred_inds or cfg.pred_inds
    #
    #         split_data, image_ids, object_inds, predicate_inds = filter_data(split, cls.hicodet, obj_inds, pred_inds,
    #                                                                          filter_empty_imgs=(split == Splits.TRAIN and cfg.filter_bg_only))
    #         assert len(split_data) == len(image_ids)
    #
    #         # Split train/val if needed
    #         if val:
    #             num_val_imgs = int(len(split_data) * cfg.val_ratio)
    #             class_splits[Splits.TRAIN] = hdsplit_class(split=Splits.TRAIN, hicodet=cls.hicodet,
    #                                                        data=split_data[:-num_val_imgs], image_ids=image_ids[:-num_val_imgs],
    #                                                        object_inds=object_inds, predicate_inds=predicate_inds)
    #             class_splits[Splits.VAL] = hdsplit_class(split=Splits.VAL, hicodet=cls.hicodet,
    #                                                      data=split_data[-num_val_imgs:], image_ids=image_ids[-num_val_imgs:],
    #                                                      object_inds=object_inds, predicate_inds=predicate_inds)
    #         else:
    #             class_splits[split] = hdsplit_class(split=split, hicodet=cls.hicodet, data=split_data, image_ids=image_ids,
    #                                                 object_inds=object_inds, predicate_inds=predicate_inds)
    #
    #     ret = [class_splits[s] for s in splits]
    #     if len(ret) == 1:
    #         ret = ret[0]
    #     return ret


def remap_box_pairs(box_pairs, box_mask):
    box_inds = np.full(box_mask.shape[0], fill_value=-1, dtype=np.int)
    box_inds[box_mask] = np.arange(np.sum(box_mask))
    box_pairs[:, 0] = box_inds[box_pairs[:, 0]]
    box_pairs[:, 1] = box_inds[box_pairs[:, 1]]
    return box_pairs


def filter_data(split, hicodet: HicoDet, obj_inds, pred_inds, inter_inds, filter_empty_imgs, filter_null_imgs):
    split_data = hicodet.split_data[split]  # type: List[HicoDetImData]
    image_ids = list(range(len(split_data)))

    if inter_inds is not None:
        assert obj_inds is None and pred_inds is None

    # Filter out unwanted predicates/object classes
    if obj_inds is None and pred_inds is None:
        final_objects_inds = list(range(hicodet.num_objects))
        final_pred_inds = list(range(hicodet.num_actions))
    else:
        obj_inds = set(obj_inds or range(hicodet.num_objects))
        pred_inds = pred_inds or list(range(hicodet.num_actions))
        inter_inds = inter_inds or list(range(hicodet.num_interactions))
        assert 0 in pred_inds
        pred_filtering_map = np.full(hicodet.num_actions, fill_value=-1, dtype=np.int)
        pred_filtering_map[pred_inds] = pred_inds
        inter_filtering_map = np.full(hicodet.num_interactions, fill_value=-1, dtype=np.int)
        inter_filtering_map[inter_inds] = inter_inds

        new_split_data = []
        for i, im_data in enumerate(split_data):
            boxes, box_classes, hois = im_data.boxes, im_data.box_classes, im_data.hois

            # Filter boxes based on object class
            box_mask = np.array([c in obj_inds for i, c in enumerate(box_classes)], dtype=bool)
            boxes = boxes[box_mask, :]
            box_classes = box_classes[box_mask]

            # Filter interactions based on action class or between removed boxes
            hois[:, [0, 2]] = remap_box_pairs(hois[:, [0, 2]], box_mask)
            hois[:, 1] = pred_filtering_map[hois[:, 1]]
            interaction_mask = np.all(hois >= 0, axis=1)
            hois = hois[interaction_mask, :]

            # Filter interactions based on interaction class
            inter_classes = hicodet.op_pair_to_interaction[box_classes[hois[:, 2]], hois[:, 1]]
            interaction_mask = (inter_filtering_map[inter_classes] >= 0)
            hois = hois[interaction_mask, :]

            new_split_data.append(HicoDetImData(filename=im_data.filename,
                                                boxes=boxes, box_classes=box_classes,
                                                hois=hois,
                                                wnet_actions=[im_data.wnet_actions[i] for i in np.flatnonzero(interaction_mask)]))

        num_inters = sum([im_data.hois.shape[0] for im_data in split_data])
        diff_num_inters = num_inters - sum([im_data.hois.shape[0] for im_data in new_split_data])
        if diff_num_inters > 0:
            print(f'{diff_num_inters}/{num_inters} interaction{" has" if diff_num_inters == 1 else "s have"} been filtered out.')
        split_data = new_split_data

        # Now we can add the person class: there won't be interactions with persons as an object if not in the initial indices, but the dataset
        # includes the class anyway because the model must always be able to predict it.
        final_objects_inds = sorted(obj_inds | {hicodet.human_class})
        final_pred_inds = sorted(pred_inds)

    if filter_empty_imgs or filter_null_imgs:  # empty = no boxes
        im_with_interactions = []
        for i, im_data in enumerate(split_data):
            empty = im_data.boxes.size == 0
            fg_hois = np.any(im_data.hois[:, 1] != hicodet.predicate_index[hicodet.null_interaction])
            if filter_empty_imgs and empty:
                continue
            if filter_null_imgs and ~fg_hois:
                continue
            im_with_interactions.append(i)
        num_old_images, num_new_images = len(image_ids), len(im_with_interactions)
        if num_new_images < num_old_images:
            print(f'Images have been discarded due to {"not having objects" if filter_empty_imgs else ""}'
                  f'{" or " if filter_empty_imgs and filter_null_imgs else ""}'
                  f'{"only having background interactions" if filter_null_imgs else ""}. '
                  f'Image index has changed (from {num_old_images} images to {num_new_images}).')
            image_ids = [image_ids[i] for i in im_with_interactions]
            split_data = [split_data[i] for i in im_with_interactions]
    assert len(split_data) == len(image_ids)
    assert image_ids == sorted(image_ids)

    return split_data, image_ids, final_objects_inds, final_pred_inds
