import datetime
import os
import pickle
import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import cfg
from lib.dataset.hicodet.hicodet_split import HicoDetSplitBuilder, HicoDetSplit, Splits
from lib.dataset.hicodet.pc_hicodet_singlehois_onehot_split import PrecomputedHicoDetSingleHOIsOnehotSplit
from lib.dataset.hicodet.pc_hicodet_singlehois_split import PrecomputedHicoDetSingleHOIsSplit
from lib.dataset.hicodet.pc_hicodet_imghois_split import PrecomputedHicoDetImgHOISplit
from lib.dataset.hicodet.pc_hicodet_split import PrecomputedHicoDetSplit
from lib.models.abstract_model import AbstractModel
from lib.models.generic_model import Prediction
from lib.stats.evaluator import Evaluator
from lib.stats.running_stats import RunningStats
from lib.stats.utils import Timer
from scripts.utils import print_params, get_all_models_by_name


class Launcher:
    # FIXME general: rename "object" in SPO triplets as "target" or something else to avoid ambiguity. Also "verb" might be better than "predicate"
    def __init__(self):
        Timer.gpu_sync = cfg.program.sync
        cfg.parse_args()

        if cfg.program.debug:
            try:  # PyCharm debugging
                print('Starting remote debugging (resume from debug server)')
                import pydevd_pycharm
                pydevd_pycharm.settrace('130.88.195.105', port=16004, stdoutToServer=True, stderrToServer=True)
                print('Remote debugging activated.')
            except:
                print('Remote debugging failed.')
                raise

        if cfg.program.load_train_final_output or cfg.program.resume:
            cfg.load()
        cfg.print()
        self.detector = None  # type: Union[None, AbstractModel]
        self.train_split, self.val_split, self.test_split = None, None, None  # type: Union[None, HicoDetSplit]
        self.curr_train_iter = 0
        self.start_epoch = 0

    def run(self):
        self.setup()
        if cfg.program.load_train_final_output:
            print('Start eval:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            all_predictions = self.evaluate()
            print('End eval:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        else:
            print('Start train:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            all_predictions = self.train()
            print('End train:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        with open(cfg.program.prediction_file, 'wb') as f:
            pickle.dump(all_predictions, f)
        print('Wrote results to %s.' % cfg.program.prediction_file)

    def setup(self):
        seed = 3 if not cfg.program.randomize else np.random.randint(1_000_000_000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print('RNG seed:', seed)

        # Data
        if cfg.program.seenf >= 0:
            pred_inds = sorted(pickle.load(open(cfg.program.active_classes_file, 'rb'))[Splits.TRAIN.value]['pred'].tolist())
            obj_inds = cfg.data.obj_inds
        else:
            # Load inds from configs. Note that these might be None after this step, which means all possible indices will be used.
            obj_inds = cfg.data.obj_inds
            pred_inds = cfg.data.pred_inds

        if cfg.opt.group:
            assert not cfg.opt.ohtrain
            train_ds_class = PrecomputedHicoDetImgHOISplit
        elif cfg.opt.ohtrain:
            train_ds_class = PrecomputedHicoDetSingleHOIsOnehotSplit
        else:
            train_ds_class = PrecomputedHicoDetSingleHOIsSplit
        self.train_split = HicoDetSplitBuilder.get_split(train_ds_class, split=Splits.TRAIN, obj_inds=obj_inds, pred_inds=pred_inds)
        self.val_split = HicoDetSplitBuilder.get_split(train_ds_class, split=Splits.VAL, obj_inds=obj_inds, pred_inds=pred_inds)
        self.test_split = HicoDetSplitBuilder.get_split(PrecomputedHicoDetSplit, split=Splits.TEST)

        # Model
        self.detector = get_all_models_by_name()[cfg.program.model](self.train_split)  # type: AbstractModel
        if torch.cuda.is_available():
            self.detector.cuda()
        else:
            print('!!!!!!!!!!!!!!!!! Running on CPU!')
        print_params(self.detector, breakdown=False)

        if cfg.program.resume:
            ckpt = torch.load(cfg.program.checkpoint_file)
            self.detector.load_state_dict(ckpt['state_dict'])
            self.start_epoch = ckpt['epoch'] + 1
            self.curr_train_iter = ckpt['curr_iter'] + 1
            print(f'Continuing from epoch {self.start_epoch} @ iteration {self.curr_train_iter}.')
        elif cfg.program.load_train_final_output:
            ckpt = torch.load(cfg.program.saved_model_file)
            self.detector.load_state_dict(ckpt['state_dict'])

    def get_optim(self):
        params = self.detector.parameters()
        if cfg.program.resume:
            params = [{'params': p, 'initial_lr': cfg.opt.lr} for p in self.detector.parameters() if p.requires_grad]

        if cfg.opt.adam:
            optimizer = torch.optim.Adam(params, weight_decay=cfg.opt.l2_coeff, lr=cfg.opt.lr, eps=1e-3)
        else:
            optimizer = torch.optim.SGD(params, weight_decay=cfg.opt.l2_coeff, lr=cfg.opt.lr, momentum=cfg.opt.momentum)

        lr_gamma = cfg.opt.lr_gamma
        if lr_gamma > 0:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.opt.lr_decay_period, gamma=lr_gamma,
                                                        last_epoch=self.start_epoch - 1)
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True, threshold_mode='abs', cooldown=1)
        return optimizer, scheduler

    def train(self):
        os.makedirs(cfg.program.output_path, exist_ok=True)

        optimizer, scheduler = self.get_optim()

        train_loader = self.train_split.get_loader(batch_size=cfg.opt.hoi_batch_size, num_workers=cfg.data.nw)
        val_loader = self.val_split.get_loader(batch_size=cfg.opt.hoi_batch_size)
        test_loader = self.test_split.get_loader(batch_size=1)

        training_stats = RunningStats(split=Splits.TRAIN, data_loader=train_loader)
        val_stats = RunningStats(split=Splits.VAL, data_loader=val_loader, history_window=len(val_loader))
        test_stats = RunningStats(split=Splits.TEST, data_loader=test_loader, history_window=len(test_loader))

        try:
            cfg.save()
            pickle.dump({Splits.TRAIN.value: {'obj': self.train_split.active_object_classes, 'pred': self.train_split.active_predicates},
                         Splits.VAL.value: {'obj': self.val_split.active_object_classes, 'pred': self.val_split.active_predicates},
                         }, open(cfg.program.ds_inds_file, 'wb'))
            if cfg.opt.num_epochs == 0:
                torch.save({'epoch': -1,
                            'curr_iter': -1,
                            'state_dict': self.detector.state_dict()},
                           cfg.program.checkpoint_file)
                self.detector.eval()
                all_predictions = self.eval_epoch(None, test_loader, test_stats)
            else:
                for epoch in range(self.start_epoch, cfg.opt.num_epochs):
                    print('Epoch %d start.' % epoch)
                    self.detector.train()
                    self.loss_epoch(epoch, train_loader, training_stats, optimizer)
                    torch.save({'epoch': epoch,
                                'curr_iter': self.curr_train_iter,
                                'state_dict': self.detector.state_dict()},
                               cfg.program.checkpoint_file)

                    self.detector.eval()
                    val_loss = self.loss_epoch(epoch, val_loader, val_stats)
                    try:
                        scheduler.step(metrics=val_loss)
                    except TypeError:
                        scheduler.step()

                    all_predictions = self.eval_epoch(epoch, test_loader, test_stats)

                    if any([pg['lr'] <= 1e-6 for pg in optimizer.param_groups]):
                        print('Exiting training early.', flush=True)
                        break
            Timer.get().print()
        finally:
            training_stats.close_tensorboard_logger()

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
            batch.epoch = epoch_idx
            batch.iter = self.curr_train_iter

            stats.batch_tic()
            batch_loss = self.loss_batch(batch, stats, optimizer)
            if optimizer is None:
                epoch_loss += batch_loss.detach()
            stats.batch_toc()

            verbose = (batch_idx % (cfg.program.print_interval * (100 if optimizer is None else 1)) == 0)
            if optimizer is not None:
                if batch_idx % cfg.program.log_interval == 0:
                    stats.log_stats(self.curr_train_iter, verbose=verbose,
                                    lr=optimizer.param_groups[0]['lr'], epoch=epoch_idx, batch=batch_idx)
                self.curr_train_iter += 1
            else:
                if verbose:
                    stats.print_times(epoch_idx, batch=batch_idx, curr_iter=self.curr_train_iter)

            # torch.cuda.empty_cache()  # Otherwise after some epochs the GPU goes out of memory. Seems to be a bug in PyTorch 0.4.1.

        if optimizer is None:
            stats.log_stats(self.curr_train_iter, epoch_idx)
        epoch_loss /= len(data_loader)
        stats.epoch_toc()
        return epoch_loss

    def loss_batch(self, batch, stats: RunningStats, optimizer=None):
        """ :arg `optimizer` should be None on validation batches. """

        losses = self.detector(batch, inference=False)
        assert losses is not None
        loss = sum(losses.values())  # type: torch.Tensor
        losses['total_loss'] = loss

        batch_stats = {'losses': losses}
        values_to_monitor = {k: v.detach().cpu() for k, v in self.detector.values_to_monitor.items()}
        if values_to_monitor:
            batch_stats['hist'] = values_to_monitor

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in self.detector.parameters() if p.grad is not None], max_norm=cfg.opt.grad_clip)

            if cfg.program.monitor:
                batch_stats['grads'] = {k + '_gradnorm': v.grad.detach().cpu().norm() for k, v in self.detector.named_parameters()
                                        if v.requires_grad and 'bias' not in k}

            optimizer.step()

        stats.update_stats(batch_stats)

        return loss

    def eval_epoch(self, epoch_idx, data_loader, stats: RunningStats):
        self.detector.eval()

        try:
            with open(cfg.program.prediction_file, 'rb') as f:
                all_predictions = pickle.load(f)
            print('Results loaded from %s.' % cfg.program.prediction_file)
        except FileNotFoundError:
            all_predictions = []

            watched_values = {}

            stats.epoch_tic()
            for batch_idx, batch in enumerate(data_loader):
                stats.batch_tic()
                prediction = self.detector(batch)  # type: Prediction
                all_predictions.append(vars(prediction))
                stats.batch_toc()

                try:
                    for k, v in self.detector.values_to_monitor.items():
                        watched_values.setdefault(k, []).append(v)
                except AttributeError:
                    pass

                if batch_idx % 20 == 0:
                    torch.cuda.empty_cache()  # Otherwise after some epochs the GPU goes out of memory. Seems to be a bug in PyTorch 0.4.1.
                if batch_idx % 1000 == 0:
                    stats.print_times(epoch_idx, batch=batch_idx, curr_iter=self.curr_train_iter)

            if watched_values:
                with open(cfg.program.watched_values_file, 'wb') as f:
                    pickle.dump(watched_values, f)

        evaluator = Evaluator(data_loader.dataset)
        evaluator.evaluate_predictions(all_predictions)
        evaluator.save(cfg.program.eval_res_file)
        metric_dicts = evaluator.output_metrics()
        if self.train_split.active_predicates.size < self.train_split.hicodet.num_predicates:
            seen_preds = sorted(self.train_split.active_predicates)
            print('Trained on:')
            _, tr_hoi_metrics = evaluator.output_metrics(actions_to_keep=seen_preds)
            tr_hoi_metrics = {f'tr_{k}': v for k, v in tr_hoi_metrics.items()}

            unseen_preds = sorted(set(range(self.train_split.hicodet.num_predicates)) - set(self.train_split.active_predicates))
            print('Zero-shot:')
            _, zs_hoi_metrics = evaluator.output_metrics(actions_to_keep=unseen_preds)
            zs_hoi_metrics = {f'zs_{k}': v for k, v in zs_hoi_metrics.items()}

            metric_dicts = list(metric_dicts) + [tr_hoi_metrics, zs_hoi_metrics]

        metrics = {k: v for md in metric_dicts for k, v in md.items()}
        stats.update_stats({'metrics': {k: np.mean(v) for k, v in metrics.items()}})
        stats.log_stats(self.curr_train_iter, epoch_idx)

        stats.epoch_toc()
        return all_predictions

    def evaluate(self):
        test_loader = self.test_split.get_loader(batch_size=1)
        test_stats = RunningStats(split=Splits.TEST, data_loader=test_loader, history_window=len(test_loader), tboard_log=False)
        all_predictions = self.eval_epoch(None, test_loader, test_stats)
        return all_predictions


def main():
    Launcher().run()


if __name__ == '__main__':
    main()
