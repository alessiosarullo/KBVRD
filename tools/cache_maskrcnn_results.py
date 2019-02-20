import sys

import h5py
import numpy as np

from config import Configs as cfg
from lib.dataset.hicodet import HicoDetSplit
from lib.dataset.minibatch import Minibatch
from lib.models.mask_rcnn import MaskRCNN
from lib.utils.data import Splits


def save_feats():
    batch_size = 1

    sys.argv += ['--batch_size', str(batch_size)]
    cfg.parse_args()

    im_inds = list(range(cfg.program.num_images)) if cfg.program.num_images > 0 else None
    hds = HicoDetSplit(Splits.TRAIN, im_inds=im_inds)
    hdsl = hds.get_loader(batch_size=batch_size, shuffle=False, drop_last=False)

    mask_rcnn = MaskRCNN()
    mask_rcnn.cuda()
    mask_rcnn.eval()

    precomputed_feats_fn = cfg.program.precomputed_feats_file_format % cfg.model.rcnn_arch
    feat_file = h5py.File(precomputed_feats_fn, 'w')
    feat_file.create_dataset('box_feats', shape=(0, mask_rcnn.output_feat_dim), maxshape=(None, mask_rcnn.output_feat_dim))
    feat_file.create_dataset('box_scores', shape=(0, hds.num_object_classes + 1), maxshape=(None, hds.num_object_classes + 1))  # FIXME magic const
    # feat_file.create_dataset('feat_maps', shape=(0, 1024, 7, 7), maxshape=(None, 1024, 7, 7))  # FIXME check

    try:
        all_boxes, box_im_ids, all_pred_classes = [], [], []
        cached_feats, cached_scores = [], []
        for im_i, im_data in enumerate(hdsl):
            im_data = im_data  # type: Minibatch

            box_class_scores, boxes, box_classes, im_ids_in_batch, masks, feat_map, box_feats, scores = mask_rcnn(im_data, return_det_results=True)
            assert np.all(im_ids_in_batch == 0)  # because batch size is 1
            im_ids = np.full((boxes.shape[0], 1), fill_value=im_i)
            pred_classes = np.stack([box_classes.astype(np.float, copy=False), box_class_scores], axis=1)
            box_feats = box_feats.cpu().numpy()

            box_im_ids.append(im_ids)
            all_boxes.append(boxes)
            all_pred_classes.append(pred_classes)

            cached_feats.append(box_feats)
            # cached_fmaps.append(feat_map)
            cached_scores.append(scores)
            if im_i % 100 == 0 or im_i == len(hdsl) - 1:
                print('Image %6d/%d' % (im_i, len(hdsl)))
                cached_feats = np.concatenate(cached_feats, axis=0)
                cached_scores = np.concatenate(cached_scores, axis=0)
                # cached_fmaps = np.concatenate(cached_fmaps, axis=0)

                num_rois, feat_dim = cached_feats.shape
                assert feat_dim == mask_rcnn.output_feat_dim
                assert num_rois == cached_scores.shape[0]

                if num_rois > 0:
                    feat_file['box_feats'].resize(feat_file['box_feats'].shape[0] + num_rois, axis=0)
                    feat_file['box_feats'][-num_rois:, :] = cached_feats

                    feat_file['box_scores'].resize(feat_file['box_scores'].shape[0] + num_rois, axis=0)
                    feat_file['box_scores'][-num_rois:, :] = cached_scores

                    # feat_file['feat_maps'].resize(feat_file['feat_maps'].shape[0] + num_rois, axis=0)
                    # feat_file['feat_maps'][-num_rois:, :] = cached_fmaps
                cached_feats, cached_fmaps, cached_scores = [], [], []

        all_boxes = np.concatenate(all_boxes, axis=0)
        box_im_ids = np.concatenate(box_im_ids)
        all_pred_classes = np.concatenate(all_pred_classes, axis=0)
        feat_file.create_dataset('boxes', data=all_boxes)
        feat_file.create_dataset('box_im_ids', data=box_im_ids.astype(np.int, copy=False))
        feat_file.create_dataset('box_pred_classes', data=all_pred_classes)
        feat_file.create_dataset('image_index', data=np.array(hds.image_index, dtype=np.int))
        assert feat_file['box_feats'].shape[0] == all_boxes.shape[0] == all_pred_classes.shape[0], \
            (feat_file['box_feats'].shape[0], all_boxes.shape[0], all_pred_classes.shape[0])
        # assert feat_file['feat_maps'].shape[0] == feat_file['image_index'].shape[0], \
        #     (feat_file['feat_maps'].shape[0], feat_file['image_index'].shape[0])
    finally:
        feat_file.close()
        print('Feat file closed.')


if __name__ == '__main__':
    save_feats()
