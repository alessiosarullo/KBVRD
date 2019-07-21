import os

import cv2
import numpy as np
from torch.utils.data import Dataset

import h5py
import torch
from config import cfg
from lib.dataset.hico.hico import Hico
from lib.dataset.utils import Splits, preprocess_img
from lib.detection.wrappers import COCO_CLASSES
from lib.dataset.hicodet.hicodet_split import Example


class HicoSplit(Dataset):
    def __init__(self, split, hico: Hico, image_ids, object_inds, predicate_inds):
        assert split in Splits

        self.split = split
        self.hico = hico  # type: Hico
        self.image_ids = image_ids

        object_inds = sorted(object_inds)
        self.objects = [hico.objects[i] for i in object_inds]
        self.active_object_classes = np.array(object_inds, dtype=np.int)
        reduced_object_index = {obj: i for i, obj in enumerate(self.objects)}
        if len(object_inds) < self.hico.num_object_classes:
            print(f'{split.value.capitalize()} objects:', object_inds)

        predicate_inds = sorted(predicate_inds)
        self.predicates = [hico.predicates[i] for i in predicate_inds]
        self.active_predicates = np.array(predicate_inds, dtype=np.int)
        reduced_predicate_index = {pred: i for i, pred in enumerate(self.predicates)}
        if len(predicate_inds) < self.hico.num_predicates:
            print(f'{split.value.capitalize()} predicates:', predicate_inds)

        reduced_interactions = np.array([[reduced_predicate_index.get(self.hico.predicates[p], -1),
                                          reduced_object_index.get(self.hico.objects[o], -1)]
                                         for p, o in self.hico.interactions])
        self.reduced_interactions = reduced_interactions[np.all(reduced_interactions >= 0, axis=1), :]  # reduced predicate and object inds
        self.active_interactions = np.array(sorted(set(np.unique(self.hico.op_pair_to_interaction[:, self.active_predicates]).tolist()) - {-1}),
                                            dtype=np.int)
        self.interactions = self.hico.interactions[self.active_interactions, :]  # original predicate and object inds

        # Checks
        interactions = np.array([[p if self.hico.predicates[p] in reduced_predicate_index else -1,
                                  o if self.hico.objects[o] in reduced_object_index else -1]
                                 for p, o in self.hico.interactions])
        assert np.all(self.interactions == interactions[np.all(interactions >= 0, axis=1), :])
        assert np.all([reduced_predicate_index[self.hico.predicates[p]] == self.reduced_interactions[i, 0] and
                       reduced_object_index[self.hico.objects[o]] == self.reduced_interactions[i, 1]
                       for i, (p, o) in enumerate(self.interactions)])

        # Compute mappings to and from COCO
        coco_obj_to_idx = {('hair dryer' if c == 'hair drier' else c).replace(' ', '_'): i for i, c in COCO_CLASSES.items()}
        assert set(coco_obj_to_idx.keys()) - {'__background__'} == set(hico.objects)
        self.hico_to_coco_mapping = np.array([coco_obj_to_idx[obj] for obj in self.objects], dtype=np.int)

        ############################
        # TODO
        split_str = (Splits.TEST if self.split == Splits.TEST else Splits.TRAIN).value
        precomputed_feats_fn = os.path.join(cfg.program.cache_root, f'precomputed__HICO__{cfg.model.rcnn_arch}_{split_str}.h5')
        print('Loading HICO precomputed feats for %s split from %s.' % (self.split.value, precomputed_feats_fn))

        pc_file = h5py.File(precomputed_feats_fn, 'r')
        self.pc_image_ids = pc_file['image_ids'][:]
        self.pc_image_infos = pc_file['img_infos'][:]
        self.pc_feats = pc_file['box_feats'][:]
        assert self.pc_feats.shape[1] == 2048  # FIXME magic constant
        # Filter HOIs. Object class filtering is currently not supported.
        if self.active_object_classes.size < self.hicodet.num_object_classes:
            raise NotImplementedError('Object class filtering is not supported.')
        if len(self.active_predicates) < self.hicodet.num_predicates and self.split != Splits.TEST:
            self.pc_action_labels = self.pc_action_labels[:, self.active_predicates]
            # Note: boxes with no interactions are NOT filtered

        # Map image IDs to indices over the precomputed image IDs
        assert len(set(self.image_ids) - set(self.pc_image_ids.tolist())) == 0
        assert len(self.pc_image_ids) == len(set(self.pc_image_ids))
        self.im_id_to_pc_im_idx = {}
        for im_id in self.image_ids:
            pc_im_idx = np.flatnonzero(self.pc_image_ids == im_id).tolist()  # type: List
            assert len(pc_im_idx) == 1, pc_im_idx
            assert im_id not in self.im_id_to_pc_im_idx
            self.im_id_to_pc_im_idx[im_id] = pc_im_idx[0]

    @property
    def precomputed_visual_feat_dim(self):
        return self.pc_feats.shape[1]

    @property
    def human_class(self) -> int:
        return self.hico.human_class

    @property
    def num_object_classes(self):
        return len(self.objects)

    @property
    def num_predicates(self):
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
        return self.hico.get_img_dir(split)

    def get_loader(self, batch_size, num_workers=0, num_gpus=1, shuffle=None, drop_last=True, **kwargs):
        if shuffle is None:
            shuffle = True if self.split == Splits.TRAIN else False
        batch_size = batch_size * num_gpus
        if self.split == Splits.TEST and batch_size > 1:
            print('! Only single-image batches are supported during prediction. Batch size changed from %d to 1.' % batch_size)
            batch_size = 1

        data_loader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda x: PrecomputedMinibatch.collate(x),
            drop_last=drop_last,
            # pin_memory=True,  # disable this in case of freezes
            **kwargs,
        )
        return data_loader

    def get_img_entry(self, idx, read_img=True) -> Example:
        img_id = self.image_ids[idx]
        img_fn = self.hico.split_filenames[self.split][img_id]
        entry = Example(idx_in_split=idx, img_id=img_id, filename=img_fn, split=self.split)
        if read_img:
            raw_image = cv2.imread(os.path.join(self.img_dir, img_fn))
            img_h, img_w = raw_image.shape[:2]
            image, img_scale_factor = preprocess_img(raw_image)
            img_size = [img_h, img_w]

            entry.image = image
            entry.img_size = img_size
            entry.scale = img_scale_factor
        return entry

    def __getitem__(self, idx):
        return self.get_img_entry(idx)

    def __len__(self):
        return self.num_images

    @classmethod
    def get_splits(cls, pred_inds=None, obj_inds=None):
        class_splits = {}
        hico = Hico()
        for split in Splits:
            if split == Splits.VAL:
                assert Splits.TRAIN not in class_splits or cfg.data.val_ratio == 0
                continue  # Val in instantiated with train, if needed

            image_ids = list(range(len(hico.split_filenames)))

            # Split train/val if needed
            if cfg.data.val_ratio > 0 and split == Splits.TRAIN:
                num_val_imgs = int(len(image_ids) * cfg.data.val_ratio)
                class_splits[Splits.TRAIN] = cls(split=Splits.TRAIN, hico=hico, image_ids=image_ids[:-num_val_imgs],
                                                 object_inds=obj_inds, predicate_inds=pred_inds)
                class_splits[Splits.VAL] = cls(split=Splits.VAL, hico=hico, image_ids=image_ids[-num_val_imgs:],
                                               object_inds=obj_inds, predicate_inds=pred_inds)
            else:
                class_splits[split] = cls(split=split, hico=hico, image_ids=image_ids, object_inds=obj_inds, predicate_inds=pred_inds)
        return class_splits
