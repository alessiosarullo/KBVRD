import os

import h5py
import numpy as np

from config import cfg
from lib.dataset.utils import Splits


def save_feats():
    for split in [Splits.TRAIN,
                  Splits.TEST,
                  ]:

        pkls_dir = cfg.program.precomputed_data_dir_format % (cfg.model.rcnn_arch, split.value)
        os.makedirs(pkls_dir)
        precomputed_feats_fn = cfg.program.precomputed_feats_file_format % (cfg.model.rcnn_arch, split.value)
        feat_file = h5py.File(precomputed_feats_fn, 'r')
        try:
            box_feats = feat_file['box_feats']
            masks = feat_file['masks']
            chunk_size = 1000
            num_boxes = box_feats.shape[0]
            for i in range(np.ceil(num_boxes / chunk_size).astype(np.int)):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, num_boxes)
                print(start, '-', end - 1)
                box_feats_chunk = box_feats[start:end, :]
                masks_chunk = masks[start:end, :]
                with open(os.path.join(pkls_dir, 'box_%d-%d.pkl' % (start, end - 1)), 'wb') as f:
                    np.savez(f, box_feats=box_feats_chunk, masks=masks_chunk)
        finally:
            feat_file.close()
            print('%s feat file closed.' % split.value)


if __name__ == '__main__':
    save_feats()
