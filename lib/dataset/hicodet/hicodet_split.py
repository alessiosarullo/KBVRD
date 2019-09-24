from typing import List

import h5py
import numpy as np

from config import cfg
from lib.containers import PrecomputedMinibatch
from lib.dataset.hicodet.hicodet import HicoDet
from lib.dataset.hoi_dataset_split import AbstractHoiDatasetSplit
from lib.dataset.utils import Splits, get_hico_to_coco_mapping


class PrecomputedFilesHandler:
    files = {}

    def __init__(self):
        super().__init__()

    @classmethod
    def get_file(cls, file_name):
        return cls.files.setdefault(file_name, {}).setdefault('_handler', h5py.File(file_name, 'r'))

    @classmethod
    def get(cls, file_name, attribute_name, load_in_memory=False):
        file = cls.get_file(file_name)
        key = attribute_name
        if load_in_memory:
            key += '__loaded'
        if key not in cls.files[file_name].keys():
            attribute = file[attribute_name]
            if load_in_memory:
                attribute = attribute[:]
            cls.files[file_name][key] = attribute
        return cls.files[file_name][key]


class HicoDetSplit(AbstractHoiDatasetSplit):
    def __init__(self, split, full_dataset: HicoDet, image_inds=None):
        assert split in Splits
        if cfg.filter_bg_only:
            raise ValueError('This option is incompatible with ROI-oriented datasets.')

        self.split = split
        self.full_dataset = full_dataset  # type: HicoDet
        self._data_split = Splits.TEST if split == Splits.TEST else Splits.TRAIN  # val -> train

        #############################################################################################################################
        # Load precomputed data
        #############################################################################################################################
        self.precomputed_feats_fn = cfg.precomputed_feats_format % ('hicodet', cfg.rcnn_arch, self._data_split.value)
        print(f'Loading precomputed feats for {self.split.value} split.')

        self.pc_image_ids = PrecomputedFilesHandler.get(self.precomputed_feats_fn, 'image_ids', load_in_memory=True)
        self.pc_image_infos = PrecomputedFilesHandler.get(self.precomputed_feats_fn, 'img_infos', load_in_memory=True)
        assert len(self.pc_image_ids) == len(set(self.pc_image_ids))

        self.pc_boxes_ext = PrecomputedFilesHandler.get(self.precomputed_feats_fn, 'boxes_ext', load_in_memory=True).astype(np.float32, copy=False)
        self.pc_boxes_feats = PrecomputedFilesHandler.get(self.precomputed_feats_fn, 'box_feats', load_in_memory=False)
        try:
            self.pc_box_labels = PrecomputedFilesHandler.get(self.precomputed_feats_fn, 'box_labels', load_in_memory=True)
        except KeyError:
            self.pc_box_labels = None

        self.pc_ho_infos = PrecomputedFilesHandler.get(self.precomputed_feats_fn, 'ho_infos', load_in_memory=True).astype(np.int)
        self.pc_union_boxes = PrecomputedFilesHandler.get(self.precomputed_feats_fn, 'union_boxes', load_in_memory=True).astype(np.float32,
                                                                                                                                copy=False)
        self.pc_union_boxes_feats = PrecomputedFilesHandler.get(self.precomputed_feats_fn, 'union_boxes_feats', load_in_memory=False)
        try:
            self.pc_action_labels = PrecomputedFilesHandler.get(self.precomputed_feats_fn, 'action_labels', load_in_memory=True)
        except KeyError:
            self.pc_action_labels = None

        # Derived
        self.pc_box_im_idxs = self.pc_boxes_ext[:, 0].astype(np.int)
        self.pc_ho_im_idxs = self.pc_ho_infos[:, 0]

        #############################################################################################################################
        # Define the remaining data
        #############################################################################################################################

        self.image_ids = image_inds or sorted(self.pc_image_ids)
        assert len(set(self.image_ids) - set(self.pc_image_ids.tolist())) == 0

        _data = self.full_dataset.split_data[self._data_split]
        self._data = [_data[i] for i in self.image_ids]

        self.objects = full_dataset.objects
        self.actions = full_dataset.actions
        self.interactions = full_dataset.interactions

        # Compute mappings to COCO
        self.hico_to_coco_mapping = get_hico_to_coco_mapping(hico_objects=full_dataset.objects)

        #############################################################################################################################
        # Cache variables to speed up loading
        #############################################################################################################################

        # Map image IDs to indices over the precomputed image IDs
        self.im_id_to_pc_im_idx = {}
        for im_id in self.image_ids:
            pc_im_idx = np.flatnonzero(self.pc_image_ids == im_id).tolist()  # type: List
            assert len(pc_im_idx) == 1, pc_im_idx
            assert im_id not in self.im_id_to_pc_im_idx
            self.im_id_to_pc_im_idx[im_id] = pc_im_idx[0]

        self.pc_im_box_range_inds = np.full((self.pc_image_ids.shape[0], 2), fill_value=-1, dtype=np.int)
        for i, pc_box_im_idx in enumerate(self.pc_box_im_idxs):
            if self.pc_im_box_range_inds[pc_box_im_idx, 0] < 0:
                self.pc_im_box_range_inds[pc_box_im_idx, 0] = i
            self.pc_im_box_range_inds[pc_box_im_idx, 1] = i
        # Check
        for i, (start, end) in enumerate(self.pc_im_box_range_inds):
            assert start >= 0 and end >= 0
            assert np.all(self.pc_box_im_idxs[start:end + 1] == i)

    @property
    def human_class(self) -> int:
        return self.full_dataset.human_class

    @property
    def num_objects(self):
        return len(self.objects)

    @property
    def num_actions(self):
        return len(self.actions)

    @property
    def num_interactions(self):
        return self.interactions.shape[0]

    @property
    def num_images(self):
        return len(self.image_ids)

    @property
    def img_dir(self):
        return self.full_dataset.get_img_dir(self._data_split)

    @property
    def precomputed_visual_feat_dim(self):
        return self.pc_boxes_feats.shape[1]

    @property
    def num_precomputed_hois(self):
        return self.pc_ho_infos.shape[0]

    def get_loader(self, batch_size, **kwargs):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
