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
from lib.dataset.hicodet import HicoDetInstanceSplit, Splits
from lib.models.base_model import BaseModel, AbstractHOIModule
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
        self.train_split = None  # type: HicoDetInstanceSplit
        self.eval_result_file = cfg.program.result_file_format % ('predcls' if cfg.program.predcls else 'sgdet')
        self.curr_train_iter = 0

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

        self.train_split = HicoDetInstanceSplit.get_split(split=Splits.TRAIN, flipping_prob=cfg.data.flip_prob)

        self.detector = BaseModel(self.train_split)
        self.detector.cuda()
        print_params(self.detector)

        if cfg.program.eval_only:
            ckpt = torch.load(cfg.program.saved_model_file)
            self.detector.load_state_dict(ckpt['state_dict'])
            # # TODO resume from checkpoint
            # if cfg.program.resume:
            # start_epoch = ckpt['epoch']
            # print("Continuing from epoch %d." % (start_epoch + 1))

    def get_optim(self):
        hoi_lr_coeff = cfg.opt.hoi_lr_coeff
        if hoi_lr_coeff != 1:
            # Lower the learning rate of HOI layers..
            red_lr_params = [p for n, p in self.detector.named_parameters() if n.startswith('hoi_branch') and p.requires_grad]
            other_params = [p for n, p in self.detector.named_parameters() if not n.startswith('hoi_branch') and p.requires_grad]
            params = [{'params': red_lr_params, 'lr': cfg.opt.learning_rate * hoi_lr_coeff}, {'params': other_params}]
            print('LR of HOI branch (%d parameters) multiplied by %f.' % (len(params[0]['params']), hoi_lr_coeff))
        else:
            params = self.detector.parameters()

        if cfg.opt.adam:
            optimizer = torch.optim.Adam(params, weight_decay=cfg.opt.l2_coeff, lr=cfg.opt.learning_rate, eps=1e-3)
        else:
            optimizer = torch.optim.SGD(params, weight_decay=cfg.opt.l2_coeff, lr=cfg.opt.learning_rate, momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True, threshold_mode='abs', cooldown=1)
        return optimizer, scheduler

    def train(self):
        os.makedirs(cfg.program.save_dir, exist_ok=True)
        val_split = HicoDetInstanceSplit.get_split(split=Splits.VAL)

        optimizer, scheduler = self.get_optim()

        train_loader = self.train_split.get_loader(batch_size=cfg.opt.batch_size)
        val_loader = val_split.get_loader(batch_size=cfg.opt.batch_size)
        training_stats = TrainingStats(split=Splits.TRAIN, data_loader=train_loader)
        val_stats = TrainingStats(split=Splits.VAL, data_loader=val_loader, history_window=len(val_loader))
        try:
            for epoch in range(cfg.opt.num_epochs):
                self.detector.train()
                self.train_epoch(epoch, train_loader, training_stats, optimizer)
                if cfg.program.save_dir is not None:
                    torch.save({'epoch': epoch,
                                'state_dict': self.detector.state_dict()},
                               cfg.program.checkpoint_file)

                self.detector.eval()
                val_loss = self.train_epoch(epoch, val_loader, val_stats)
                scheduler.step(val_loss)
                if any([pg['lr'] <= 1e-6 for pg in optimizer.param_groups]):  # FIXME magic constant
                    print('Exiting training early.', flush=True)
                    break
            cfg.save()
            Timer.get().print()
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

    def train_epoch(self, epoch_idx, data_loader, stats: TrainingStats, optimizer=None):
        stats.epoch_tic()
        epoch_loss = 0
        for batch_idx, batch in enumerate(data_loader):
            stats.batch_tic()
            epoch_loss += self.train_batch(batch, stats, optimizer)
            stats.batch_toc()

            self.curr_train_iter += 0 if optimizer is None else 1
            if optimizer is not None and batch_idx % cfg.program.log_interval == 0:
                stats.log_stats(self.curr_train_iter, epoch_idx, batch=batch_idx,
                                verbose=batch_idx % cfg.program.log_interval == 0,
                                lr=optimizer.param_groups[0]['lr'])  # TODO lr for each parameter group
        if optimizer is None:
            stats.log_stats(self.curr_train_iter, epoch_idx)
        epoch_loss /= len(data_loader)
        stats.epoch_toc()
        return epoch_loss

    def train_batch(self, batch, stats, optimizer=None):
        """ :arg `optimizer` should be None on validation batches. """

        losses = self.detector.get_losses(batch)
        assert losses is not None
        loss = sum(losses.values())  # type: torch.Tensor
        losses['total_loss'] = loss

        hoi_branch = self.detector.hoi_branch  # type: AbstractHOIModule
        batch_stats = {'losses': losses, 'hist': {k: v.detach().cpu() for k, v in hoi_branch.values_to_monitor.items()}}
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in self.detector.parameters() if p.grad is not None], max_norm=cfg.opt.grad_clip)

            batch_stats['watch'] = {k + '_gradnorm': v.grad.detach().cpu().norm() for k, v in hoi_branch.named_parameters() if v.requires_grad}

            optimizer.step()

        stats.update_stats(batch_stats)

        return loss

    def test(self):
        test_split = HicoDetInstanceSplit.get_split(split=Splits.TEST)
        result_file = self.eval_result_file
        try:
            with open(result_file, 'rb') as f:
                all_predictions = pickle.load(f)
            print('Loaded predictions from %s.' % result_file)
        except FileNotFoundError:
            self.detector.eval()
            test_loader = test_split.get_loader(batch_size=1)
            all_predictions = []
            Timer.get('Test').tic()
            for b_idx, batch in enumerate(test_loader):
                Timer.get('Test', 'Img').tic()
                prediction = self.detector(batch)  # type: Prediction
                all_predictions.append(vars(prediction))
                Timer.get('Test', 'Img').toc()
                if b_idx % cfg.program.print_interval == 0:
                    time_per_batch = Timer.get('Test', 'Img', get_only=True).spent(average=True)
                    time_to_load = Timer.get('GetBatch', get_only=True).spent(average=True)
                    time_to_collate = Timer.get('Collate', get_only=True).spent(average=True)
                    est_time_per_epoch = len(test_loader) * (time_per_batch + time_to_load * 1 + time_to_collate)
                    print('Img {:5d}/{:5d}.'.format(b_idx, len(test_loader)),
                          'Avg: {:>5s}/detection, {:>5s}/load, {:>5s}/collate.'.format(Timer.format(time_per_batch),
                                                                                       Timer.format(time_to_load),
                                                                                       Timer.format(time_to_collate)),
                          'Progress: {:>7s}/{:>7s} (estimated).'.format(Timer.format(Timer.get('Test', get_only=True).progress()),
                                                                        Timer.format(est_time_per_epoch)))

                    torch.cuda.empty_cache()  # Otherwise after some epochs the GPU goes out of memory. Seems to be a bug in PyTorch 0.4.1.
            Timer.get('Test').toc()

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
