import datetime
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import cfg
from lib.dataset.hicodet import HicoDetInstanceSplit, Splits
from lib.models.abstract_model import AbstractHOIBranch
from lib.models.generic_model import GenericModel, Prediction
from lib.stats.evaluator import Evaluator
from lib.stats.running_stats import RunningStats
from lib.stats.utils import Timer
from scripts.utils import print_params, get_all_models_by_name


class Launcher:
    # FIXME general: rename "object" in SPO triplets as "target" or something else to avoid ambiguity. Also "verb" might be better than "predicate"
    def __init__(self):
        Timer.gpu_sync = cfg.program.sync
        cfg.parse_args()
        if cfg.program.load_train_output:
            cfg.load()
        cfg.print()
        self.detector = None  # type: GenericModel
        self.train_split, self.val_split, self.test_split = None, None, None  # type: HicoDetInstanceSplit
        self.curr_train_iter = 0

    def run(self):
        self.setup()
        if cfg.program.load_train_output:
            print('Start eval:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            all_predictions = self.evaluate()
            print('End eval:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        else:
            print('Start train:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            all_predictions = self.train()
            print('End train:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        with open(cfg.program.result_file, 'wb') as f:
            pickle.dump(all_predictions, f)
        print('Wrote results to %s.' % cfg.program.result_file)

    def setup(self):
        seed = 3 if not cfg.program.randomize else np.random.randint(1_000_000_000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print('RNG seed:', seed)

        self.train_split = HicoDetInstanceSplit.get_split(split=Splits.TRAIN, flipping_prob=cfg.data.flip_prob)
        self.val_split = HicoDetInstanceSplit.get_split(split=Splits.VAL)
        self.test_split = HicoDetInstanceSplit.get_split(split=Splits.TEST)

        self.detector = get_all_models_by_name()[cfg.program.model](self.train_split)  # type: GenericModel
        if torch.cuda.is_available():
            self.detector.cuda()
        else:
            print('!!!!!!!!!!!!!!!!! Running on CPU!')
        print_params(self.detector, breakdown=False)

        if cfg.program.load_train_output:
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
            optimizer = torch.optim.SGD(params, weight_decay=cfg.opt.l2_coeff, lr=cfg.opt.learning_rate, momentum=cfg.opt.momentum)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True, threshold_mode='abs', cooldown=1)
        return optimizer, scheduler

    def train(self):
        os.makedirs(cfg.program.output_path, exist_ok=True)

        optimizer, scheduler = self.get_optim()

        train_loader = self.train_split.get_loader(batch_size=cfg.opt.batch_size)
        val_loader = self.val_split.get_loader(batch_size=cfg.opt.batch_size if not cfg.program.model.startswith('nmotifs') else 1)  # FIXME?
        test_loader = self.test_split.get_loader(batch_size=1)

        training_stats = RunningStats(split=Splits.TRAIN, data_loader=train_loader)
        val_stats = RunningStats(split=Splits.VAL, data_loader=val_loader, history_window=len(val_loader))
        test_stats = RunningStats(split=Splits.TEST, data_loader=test_loader, history_window=len(test_loader))

        try:
            for epoch in range(cfg.opt.num_epochs):
                print('Epoch %d start.' % epoch)
                self.detector.train()
                self.loss_epoch(epoch, train_loader, training_stats, optimizer)
                torch.save({'epoch': epoch,
                            'state_dict': self.detector.state_dict()},
                           cfg.program.checkpoint_file)

                self.detector.eval()
                val_loss = self.loss_epoch(epoch, val_loader, val_stats)
                scheduler.step(val_loss)

                all_predictions = self.eval_epoch(epoch, test_loader, test_stats)

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
            os.rename(cfg.program.checkpoint_file, cfg.program.saved_model_file)

        # noinspection PyUnboundLocalVariable
        return all_predictions

    def loss_epoch(self, epoch_idx, data_loader, stats: RunningStats, optimizer=None):
        stats.epoch_tic()
        epoch_loss = 0
        for batch_idx, batch in enumerate(data_loader):
            stats.batch_tic()
            epoch_loss += self.loss_batch(batch, stats, optimizer)
            stats.batch_toc()

            verbose = (batch_idx % (cfg.program.print_interval * (100 if optimizer is None else 1)) == 0)
            if optimizer is not None:
                if batch_idx % cfg.program.log_interval == 0:
                    stats.log_stats(self.curr_train_iter, verbose=verbose,
                                    lr=optimizer.param_groups[0]['lr'])  # TODO lr for each parameter group
                self.curr_train_iter += 1
            else:
                if verbose:
                    stats.print_times(epoch_idx, batch=batch_idx, curr_iter=self.curr_train_iter)

        if optimizer is None:
            stats.log_stats(self.curr_train_iter, epoch_idx)
        epoch_loss /= len(data_loader)
        stats.epoch_toc()
        return epoch_loss

    def loss_batch(self, batch, stats, optimizer=None):
        """ :arg `optimizer` should be None on validation batches. """

        losses = self.detector.get_losses(batch)
        assert losses is not None
        loss = sum(losses.values())  # type: torch.Tensor
        losses['total_loss'] = loss

        hoi_branch = self.detector.hoi_branch  # type: AbstractHOIBranch
        batch_stats = {'losses': losses,
                       # 'hist': {k: v.detach().cpu() for k, v in hoi_branch.values_to_monitor.items()}
                       }
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in self.detector.parameters() if p.grad is not None], max_norm=cfg.opt.grad_clip)

            # batch_stats['watch'] = {k + '_gradnorm': v.grad.detach().cpu().norm() for k, v in hoi_branch.named_parameters() if v.requires_grad}

            optimizer.step()

        stats.update_stats(batch_stats)

        return loss

    def eval_epoch(self, epoch_idx, data_loader, stats: RunningStats):
        self.detector.eval()
        all_predictions = []

        stats.epoch_tic()
        for batch_idx, batch in enumerate(data_loader):
            stats.batch_tic()
            prediction = self.detector(batch)  # type: Prediction
            all_predictions.append(vars(prediction))
            stats.batch_toc()

            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()  # Otherwise after some epochs the GPU goes out of memory. Seems to be a bug in PyTorch 0.4.1.
            if batch_idx % 1000 == 0:
                stats.print_times(epoch_idx, batch=batch_idx, curr_iter=self.curr_train_iter)

        evaluator = Evaluator.evaluate_predictions(data_loader.dataset, all_predictions)  # type: Evaluator
        evaluator.print_metrics()
        stats.update_stats({'metrics': {k: np.mean(v) for k, v in evaluator.metrics.items()}})
        stats.log_stats(self.curr_train_iter, epoch_idx)

        stats.epoch_toc()
        return all_predictions

    def evaluate(self):
        test_loader = self.test_split.get_loader(batch_size=1)
        test_stats = RunningStats(split=Splits.TEST, data_loader=test_loader, history_window=len(test_loader))
        all_predictions = self.eval_epoch(None, test_loader, test_stats)
        return all_predictions


def main():
    Launcher().run()


if __name__ == '__main__':
    main()
