import datetime
import os
import random
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import Configs as cfg
from lib.dataset.hicodet import HicoDetSplit, Splits
from lib.models.base_model import BaseModel
from lib.evaluator import Evaluator
from scripts.utils import Timer
from scripts.utils import print_params


class Launcher:
    def __init__(self):
        Timer.gpu_sync = cfg.program.sync
        cfg.parse_args()
        cfg.print()
        self.detector = None  # type: BaseModel
        self.train_split = None  # type: HicoDetSplit

    def run(self):
        self.setup()
        if not cfg.program.eval_only:
            print('Start train:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            self.train()
            print('End train:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print('Start eval:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.test()
        print('End eval:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def setup(self):
        seed = 3 if not cfg.program.randomize else np.random.randint(1_000_000_000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print('RNG seed:', seed)

        self.train_split = HicoDetSplit(Splits.TRAIN, im_inds=cfg.program.im_inds, flipping_prob=cfg.data.flip_prob)

        self.detector = BaseModel(self.train_split)
        self.detector.cuda()
        print_params(self.detector)

        if cfg.program.eval_only:
            ckpt = torch.load(cfg.program.saved_model_file)
            self.detector.load_state_dict(ckpt['state_dict'])
            self.detector.mask_rcnn._load_weights()  # FIXME this is only needed because BoxHead is trained by mistake. Remove after fix
            # # TODO
            # if cfg.program.resume:
            # start_epoch = ckpt['epoch']
            # print("Continuing from epoch %d." % (start_epoch + 1))

    def get_optim(self):
        # TODO tip. Erase if unnecessary
        # Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps stabilize the models.
        params = self.detector.parameters()
        if cfg.opt.use_adam:
            optimizer = torch.optim.Adam(params, weight_decay=cfg.opt.l2_coeff, lr=cfg.opt.learning_rate, eps=1e-3)
        else:
            optimizer = torch.optim.SGD(params, weight_decay=cfg.opt.l2_coeff, lr=cfg.opt.learning_rate, momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1, verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
        return optimizer, scheduler

    def train(self):
        train_loader = self.train_split.get_loader(batch_size=cfg.opt.batch_size)

        optimizer, scheduler = self.get_optim()
        # TODO scheduler is unused so far
        for epoch in range(cfg.opt.num_epochs):
            self.detector.train()
            self.train_epoch(epoch, train_loader, optimizer)
            if cfg.program.save_dir is not None:
                torch.save({
                    'epoch': epoch,
                    'state_dict': self.detector.state_dict(),
                }, cfg.program.checkpoint_file)

        Timer.get().print()
        if cfg.opt.num_epochs > 0:
            os.remove(cfg.program.saved_model_file)
            os.symlink(os.path.abspath(cfg.program.checkpoint_file), cfg.program.saved_model_file)

    def train_epoch(self, epoch_num, train_loader, optimizer):
        tr = []
        num_batches = len(train_loader)

        Timer.get('Epoch').tic()
        for bidx, batch in enumerate(train_loader):
            Timer.get('Epoch', 'Batch').tic()
            tr.append(self.train_batch(batch, optimizer))
            Timer.get('Epoch', 'Batch').toc()

            if bidx % cfg.program.print_interval == 0:
                iter_num = num_batches * epoch_num + bidx
                time_per_batch = Timer.get('Epoch', 'Batch', get_only=True).spent(average=True)
                time_to_load = Timer.get('Epoch', 'GetBatch', get_only=True).spent(average=True)
                time_to_collate = Timer.get('Epoch', 'Collate', get_only=True).spent(average=True)
                est_time_per_epoch = num_batches * (time_per_batch + time_to_load * train_loader.batch_size + time_to_collate)

                print('Iter {:6d} (epoch {:2d}, batch {:5d}/{:5d}).'.format(iter_num, epoch_num, bidx, num_batches),
                      'Avg: {:s}/batch, {:s}/load, {:s}/collate.'.format(Timer.format(time_per_batch),
                                                                         Timer.format(time_to_load),
                                                                         Timer.format(time_to_collate)),
                      'Estimated {:s}/epoch'.format(Timer.format(est_time_per_epoch)))
                print(pd.concat(tr[-cfg.program.print_interval:], axis=1).mean(1))
                print('-' * 10, flush=True)
        Timer.get('Epoch').toc()
        print('Time for epoch:', Timer.format(Timer.get('Epoch').last))
        print('-' * 100, flush=True)

    def train_batch(self, batch, optimizer):
        losses = self.detector.get_losses(batch)
        optimizer.zero_grad()

        assert losses is not None
        loss = sum(losses.values())  # type: torch.Tensor
        loss.backward()
        losses['total'] = loss
        res = pd.Series({x: y.item() for x, y in losses.items()})

        nn.utils.clip_grad_norm_([p for p in self.detector.parameters() if p.grad is not None], max_norm=cfg.opt.grad_clip)
        optimizer.step()
        return res

    def test(self):
        test_split = HicoDetSplit(Splits.TEST, im_inds=cfg.program.im_inds)
        test_loader = test_split.get_loader(batch_size=1)  # TODO? Support larger batches

        # TODO remove if not useful.
        # In GPNN code:
        # if sequence_ids[0] is 'HICO_test2015_00000396':
        # break

        all_pred_entries = []
        evaluator = Evaluator()
        self.detector.eval()
        for b_idx, batch in enumerate(test_loader):
            Timer.get('Img').tic()
            prediction = self.detector(batch)
            Timer.get('Img').toc()

            all_pred_entries.append(prediction)
            evaluator.evaluate_scene_graph_entry(batch, prediction)

            if b_idx % cfg.program.print_interval == 0 and b_idx >= cfg.program.print_interval:
                time_per_batch = Timer.get('Img').spent(average=True)
                print("Img {:5d}/{:5d}. Avg: {:.3f}s/img".format(b_idx, len(test_loader), time_per_batch))

        evaluator.print_stats()
        with open(cfg.program.result_file, 'wb') as f:
            pickle.dump(all_pred_entries, f)
        print('Wrote results to %s. Terminating.' % cfg.program.result_file)


def main():
    Launcher().run()


if __name__ == '__main__':
    main()
