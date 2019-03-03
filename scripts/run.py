import datetime
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import Configs as cfg
from lib.containers import Prediction
from lib.dataset.hicodet import HicoDetInstance, Splits
from lib.models.base_model import BaseModel
from lib.stats.eval_stats import EvalStats
from lib.stats.training_stats import TrainingStats
from scripts.utils import Timer
from scripts.utils import print_params


class Launcher:
    def __init__(self):
        Timer.gpu_sync = cfg.program.sync
        cfg.parse_args()
        if cfg.program.eval_only:
            cfg.load()
            cfg.program.eval_only = True
        cfg.print()
        self.detector = None  # type: BaseModel
        self.train_split = None  # type: HicoDetInstance
        self.eval_result_file = cfg.program.result_file_format % ('predcls' if cfg.program.predcls else 'sgdet')

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

        self.train_split = HicoDetInstance(Splits.TRAIN, flipping_prob=cfg.data.flip_prob)

        self.detector = BaseModel(self.train_split)
        self.detector.cuda()
        print_params(self.detector)

        if cfg.program.eval_only:
            ckpt = torch.load(cfg.program.saved_model_file)
            self.detector.load_state_dict(ckpt['state_dict'])
            # self.detector.mask_rcnn._load_weights()  # FIXME this is only needed because BoxHead is trained by mistake. Remove after fix
            # # TODO
            # if cfg.program.resume:
            # start_epoch = ckpt['epoch']
            # print("Continuing from epoch %d." % (start_epoch + 1))

    def get_optim(self):
        # Lower the learning rate of some layers. It's a hack, but it helps stabilize the models.
        red_lr_params = [p for n, p in self.detector.named_parameters() if n.startswith('hoi_branch') and p.requires_grad]
        other_params = [p for n, p in self.detector.named_parameters() if not n.startswith('hoi_branch') and p.requires_grad]
        params = [{'params': red_lr_params, 'lr': cfg.opt.learning_rate * 10.0}, {'params': other_params}]
        print('Reduced LR of %d parameters.' % len(params[0]['params']))
        # params = self.detector.parameters()

        if cfg.opt.adam:
            optimizer = torch.optim.Adam(params, weight_decay=cfg.opt.l2_coeff, lr=cfg.opt.learning_rate, eps=1e-3)
        else:
            optimizer = torch.optim.SGD(params, weight_decay=cfg.opt.l2_coeff, lr=cfg.opt.learning_rate, momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1, verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
        return optimizer, scheduler

    def train(self):
        os.makedirs(cfg.program.save_dir, exist_ok=True)
        train_loader = self.train_split.get_loader(batch_size=cfg.opt.batch_size)

        optimizer, scheduler = self.get_optim()
        # TODO scheduler is unused so far

        training_stats = TrainingStats(num_batches=len(train_loader))
        try:
            for epoch in range(cfg.opt.num_epochs):
                self.detector.train()
                self.train_epoch(epoch, train_loader, optimizer, training_stats)
                if cfg.program.save_dir is not None:
                    torch.save({'epoch': epoch,
                                'state_dict': self.detector.state_dict()},
                               cfg.program.checkpoint_file)
            Timer.get().print()
            cfg.save()
        finally:
            training_stats.close_tensorboard_logger()

        if cfg.opt.num_epochs > 0:
            try:
                os.remove(cfg.program.saved_model_file)
            except FileNotFoundError:
                pass

            os.symlink(os.path.abspath(cfg.program.checkpoint_file), cfg.program.saved_model_file)
        try:
            os.remove(self.eval_result_file)
        except FileNotFoundError:
            pass

    def train_epoch(self, epoch_num, train_loader, optimizer, training_stats):
        num_batches = len(train_loader)

        Timer.get('Epoch').tic()
        for bidx, batch in enumerate(train_loader):
            Timer.get('Epoch', 'Batch').tic()
            self.train_batch(batch, optimizer, training_stats)
            Timer.get('Epoch', 'Batch').toc()

            if bidx % cfg.program.print_interval == 0:
                iter_num = num_batches * epoch_num + bidx
                time_per_batch = Timer.get('Epoch', 'Batch', get_only=True).spent(average=True)
                time_to_load = Timer.get('Epoch', 'GetBatch', get_only=True).spent(average=True)
                time_to_collate = Timer.get('Epoch', 'Collate', get_only=True).spent(average=True)
                est_time_per_epoch = num_batches * (time_per_batch + time_to_load * train_loader.batch_size + time_to_collate)

                print('Iter {:6d} (epoch {:2d}, batch {:5d}/{:5d}).'.format(iter_num, epoch_num, bidx, num_batches),
                      'Avg: {:>5s}/batch, {:>5s}/load, {:>5s}/collate.'.format(Timer.format(time_per_batch),
                                                                               Timer.format(time_to_load),
                                                                               Timer.format(time_to_collate)),
                      'Current epoch progress: {:>7s}/{:>7s} (estimated).'.format(Timer.format(Timer.get('Epoch', get_only=True).progress()),
                                                                                  Timer.format(est_time_per_epoch)))
                training_stats.log_stats(curr_iter=iter_num, lr=optimizer.param_groups[0]['lr'])
                print('-' * 10, flush=True)
        Timer.get('Epoch').toc()
        print('Time for epoch:', Timer.format(Timer.get('Epoch').last))
        print('-' * 100, flush=True)

    def train_batch(self, batch, optimizer, training_stats):
        losses = self.detector.get_losses(batch)
        optimizer.zero_grad()

        assert losses is not None
        loss = sum(losses.values())  # type: torch.Tensor
        loss.backward()

        losses['total_loss'] = loss
        training_stats.update_stats({'losses': losses, 'watch': self.detector.hoi_branch.last_feats})

        nn.utils.clip_grad_norm_([p for p in self.detector.parameters() if p.grad is not None], max_norm=cfg.opt.grad_clip)
        optimizer.step()

    def test(self):
        test_split = HicoDetInstance(Splits.TEST)
        result_file = self.eval_result_file
        try:
            with open(result_file, 'rb') as f:
                all_predictions = pickle.load(f)
            print('Loaded predictions from %s.' % result_file)
        except FileNotFoundError:
            self.detector.eval()
            test_loader = test_split.get_loader(batch_size=1)
            all_predictions = []
            Timer.get('Epoch').tic()
            for b_idx, batch in enumerate(test_loader):
                Timer.get('Epoch', 'Img').tic()
                prediction = self.detector(batch)  # type: Prediction
                all_predictions.append(vars(prediction))
                Timer.get('Epoch', 'Img').toc()
                if b_idx % cfg.program.print_interval == 0:
                    time_per_batch = Timer.get('Epoch', 'Img', get_only=True).spent(average=True)
                    time_to_load = Timer.get('Epoch', 'GetBatch', get_only=True).spent(average=True)
                    time_to_collate = Timer.get('Epoch', 'Collate', get_only=True).spent(average=True)
                    est_time_per_epoch = len(test_loader) * (time_per_batch + time_to_load * 1 + time_to_collate)
                    print('Img {:5d}/{:5d}.'.format(b_idx, len(test_loader)),
                          'Avg: {:>5s}/detection, {:>5s}/load, {:>5s}/collate.'.format(Timer.format(time_per_batch),
                                                                                       Timer.format(time_to_load),
                                                                                       Timer.format(time_to_collate)),
                          'Progress: {:>7s}/{:>7s} (estimated).'.format(Timer.format(Timer.get('Epoch', get_only=True).progress()),
                                                                        Timer.format(est_time_per_epoch)))

                    torch.cuda.empty_cache()  # Otherwise after some epochs the GPU goes out of memory. Seems to be a bug in PyTorch 0.4.1.
            Timer.get('Epoch').toc()

            with open(result_file, 'wb') as f:
                pickle.dump(all_predictions, f)
            print('Wrote results to %s.' % result_file)

        Timer.get('Eval').tic()
        result_stats = EvalStats.evaluate_predictions(test_split, all_predictions)
        Timer.get('Eval').toc()
        result_stats.print()


def main():
    Launcher().run()


if __name__ == '__main__':
    main()
