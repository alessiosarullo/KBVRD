"""
Training script for scene graph detection. Integrated with my faster rcnn setup
"""

import datetime
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lib.dataset.hicodet import HicoDetSplit, Splits
from config import Configs as cfg
from lib.models.base_model import BaseModel
from scripts.utils import print_para


class Trainer:
    def __init__(self):
        pass

    def train(self):
        print('Start train:', datetime.datetime.now())
        cfg.parse_args()

        detector, train_loaders = self.setup()
        print(print_para(detector), flush=True)
        detector.cuda()

        print("Training starts now!")
        optimizer, scheduler = self.get_optim(detector)
        for epoch in range(cfg.opt.num_epochs):
            self.train_epoch(epoch, train_loaders, optimizer, detector)
            if cfg.program.save_dir is not None:
                save_file = os.path.join(cfg.program.save_dir, 'ckpt.tar')
                torch.save({
                    'epoch': epoch,
                    'state_dict': detector.state_dict(),
                }, save_file)

        if cfg.program.save_dir is not None and cfg.opt.num_epochs > 0:
            os.symlink(save_file, os.path.join(cfg.program.save_dir, 'final.tar'))
        print('End train:', datetime.datetime.now())

    def setup(self):
        seed = 3 if not cfg.program.randomize else np.random.randint(1_000_000_000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print('RNG seed:', seed)

        train = HicoDetSplit(Splits.TRAIN, im_inds=list(range(8)))  # FIXME
        detector = BaseModel(train)
        train_loaders = train.get_loader(batch_size=1)

        return detector, train_loaders

    def get_optim(self, detector):
        conf = cfg.opt
        lr = conf.learning_rate

        # TODO tip. Erase if unnecessary
        # Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps stabilize the models.

        params = detector.parameters()
        if conf.use_adam:
            optimizer = torch.optim.Adam(params, weight_decay=conf.l2_coeff, lr=lr, eps=1e-3)
        else:
            optimizer = torch.optim.SGD(params, weight_decay=conf.l2_coeff, lr=lr, momentum=0.9)

        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1, verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
        return optimizer, scheduler

    def train_epoch(self, epoch_num, train_loaders, optimizer, detector):
        print_interval = cfg.program.print_interval
        for train_split_idx, train_loader in enumerate(train_loaders):
            tr = []
            print('Using train split %d.' % train_split_idx)
            detector.set_training_mode(param_set_id=train_split_idx)
            num_batches = len(train_loader)

            start_time = time.time()
            for b, batch in enumerate(train_loader):
                batch.iter_num = num_batches * epoch_num + b
                tr.append(self.train_batch(batch, optimizer, detector))

                if b % print_interval == 0 and b >= print_interval:
                    time_per_batch = (time.time() - start_time) / (b + 1)
                    print("\nIter {:6d} (epoch {:2d}, batch {:5d}/{:5d}). {:.3f}s/batch, {:.1f}m/epoch".format(
                        batch.iter_num, epoch_num, b, num_batches, time_per_batch, num_batches * time_per_batch / 60))
                    print(pd.concat(tr[-print_interval:], axis=1).mean(1))
                    print('-----------', flush=True)
            print('Time for epoch %d, train split %d: %.0fm' % (epoch_num, train_split_idx, (time.time() - start_time) / 60))

    def train_batch(self, b, optimizer, detector: BaseModel):
        losses = detector.get_losses(b)
        optimizer.zero_grad()

        assert losses is not None
        loss = sum(losses.values())
        loss.backward()
        losses['total'] = loss
        res = pd.Series({x: y.data[0] for x, y in losses.items()})

        nn.utils.clip_grad_norm([(n, p) for n, p in detector.named_parameters() if p.grad is not None], max_norm=cfg.opt.grad_clip)
        optimizer.step()
        return res


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
