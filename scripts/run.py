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

    def run(self):
        detector, train_split = self.setup()
        if not cfg.program.eval_only:
            print('Start train:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            self.train(detector, train_split)
            print('End train:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print('Start eval:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.test(detector)
        print('End eval:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def train(self, detector, train_split):
        train_loader = train_split.get_loader(batch_size=cfg.opt.batch_size)

        optimizer, scheduler = self.get_optim(detector)
        for epoch in range(cfg.opt.num_epochs):
            detector.train()
            self.train_epoch(epoch, train_loader, optimizer, detector)
            if cfg.program.save_dir is not None:
                torch.save({
                    'epoch': epoch,
                    'state_dict': detector.state_dict(),
                }, cfg.program.checkpoint_file)

        Timer.get().print()
        if cfg.opt.num_epochs > 0:
            link = os.path.join(cfg.program.save_dir, 'final.tar')
            os.remove(link)
            os.symlink(os.path.abspath(cfg.program.checkpoint_file), link)

    def test(self, detector: BaseModel):
        test_split = HicoDetSplit(Splits.TEST, im_inds=cfg.program.im_inds)
        test_loader = test_split.get_loader(batch_size=1)  # TODO? Support larger batches

        # TODO remove if not useful.
        # In GPNN code:
        # if sequence_ids[0] is 'HICO_test2015_00000396':
            #break

        all_pred_entries = []
        evaluator = Evaluator()
        detector.set_eval_mode()
        for batch in test_loader:
            prediction = detector(batch)
            all_pred_entries.append(prediction)
            assert len(prediction.obj_im_inds.unique()) == len(np.unique(prediction.hoi_img_inds)) == 1
            evaluator.evaluate_scene_graph_entry(batch, prediction)
        evaluator.print_stats()

        res_file = os.path.join(cfg.program.save_dir, 'result_test.pkl')
        with open(res_file, 'wb') as f:
            pickle.dump(all_pred_entries, f)
        print('Wrote results to %s. Terminating.' % res_file)

    @staticmethod
    def setup():
        seed = 3 if not cfg.program.randomize else np.random.randint(1_000_000_000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print('RNG seed:', seed)

        train_split = HicoDetSplit(Splits.TRAIN, im_inds=cfg.program.im_inds, flipping_prob=cfg.data.flip_prob)

        detector = BaseModel(train_split)
        detector.cuda()
        print_params(detector)

        return detector, train_split

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

    def train_epoch(self, epoch_num, train_loader, optimizer, detector):
        print_interval = cfg.program.print_interval
        tr = []
        num_batches = len(train_loader)

        Timer.get('Epoch').tic()
        for b, batch in enumerate(train_loader):
            Timer.get('Epoch', 'Batch').tic()
            batch.iter_num = num_batches * epoch_num + b
            tr.append(self.train_batch(batch, optimizer, detector))
            Timer.get('Epoch', 'Batch').toc()

            if b % print_interval == 0 and b >= print_interval:
                time_per_batch = Timer.get('Epoch', 'Batch').spent(average=True)
                print("Iter {:6d} (epoch {:2d}, batch {:5d}/{:5d}). {:.3f}s/batch, {:.1f}m/epoch".format(
                    batch.iter_num, epoch_num, b, num_batches, time_per_batch, num_batches * time_per_batch / 60))
                print(pd.concat(tr[-print_interval:], axis=1).mean(1))
                print('-' * 10, flush=True)
        Timer.get('Epoch').toc()
        print('Time for epoch:', Timer.get('Epoch').str_last())
        print('-' * 100, flush=True)

    @staticmethod
    def train_batch(b, optimizer, detector: BaseModel):
        losses = detector.get_losses(b)
        optimizer.zero_grad()

        assert losses is not None
        loss = sum(losses.values())  # type: torch.Tensor
        loss.backward()
        losses['total'] = loss
        res = pd.Series({x: y.item() for x, y in losses.items()})

        nn.utils.clip_grad_norm_([p for p in detector.parameters() if p.grad is not None], max_norm=cfg.opt.grad_clip)
        optimizer.step()
        return res


def main():
    Launcher().run()


if __name__ == '__main__':
    main()
