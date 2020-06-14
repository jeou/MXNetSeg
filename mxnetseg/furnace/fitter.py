# coding=utf-8

import os
import time
import platform
import fitlog
from tqdm import tqdm
from mxnet import nd, autograd
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader
from gluoncv.utils import LRScheduler, split_and_load
from gluoncv.utils.metrics import SegmentationMetric
from mxnetseg.models import SegBaseModel
from mxnetseg.data import segmentation_dataset
from mxnetseg.tools import image_transform, get_strftime, get_logger, cudnn_auto_tune


def _criterion(aux, aux_weight, ohem=False):
    """return loss function"""
    if ohem:
        from gluoncv.loss import MixSoftmaxCrossEntropyOHEMLoss
        return MixSoftmaxCrossEntropyOHEMLoss(aux=aux, aux_weight=aux_weight, ignore_label=-1)
    else:
        # from gluoncv.loss import MixSoftmaxCrossEntropyLoss
        from .loss import MultiSoftmaxCrossEntropyLoss
        return MultiSoftmaxCrossEntropyLoss(aux=aux, aux_weight=aux_weight, ignore_label=-1)


def _data_iter(data_name: str, batch_size: int, shuffle: bool = False,
               last_batch: str = 'discard', **kwargs):
    """return data iter"""
    transformer = image_transform()
    num_worker = 0 if platform.system() == 'Windows' else 16
    data_set = segmentation_dataset(data_name, transform=transformer, **kwargs)
    data_iter = DataLoader(data_set, batch_size, shuffle, last_batch=last_batch,
                           num_workers=num_worker)
    return data_iter, len(data_set)


def _adam_trainer(net: SegBaseModel, lr: float, wd: float, beta1: float, beta2: float):
    """return adam trainer"""
    trainer = Trainer(net.collect_params(), 'adam',
                      optimizer_params={'learning_rate': lr, 'wd': wd, 'beta1': beta1, 'beta2': beta2})
    return trainer


def _trainer(net: SegBaseModel, optimizer: str, lr: float, lr_scheduler: str,
             wd: float, momentum: float = 0.9, **kwargs):
    """return sgd/nag trainer"""
    scheduler = _lr_scheduler(lr_scheduler, lr, **kwargs)
    trainer = Trainer(net.collect_params(), optimizer,
                      optimizer_params={'lr_scheduler': scheduler, 'wd': wd, 'momentum': momentum,
                                        'multi_precision': True})
    return trainer


def _lr_scheduler(mode: str, base_lr: float = 0.1, target_lr: float = .0, nepochs: int = 0,
                  iters_per_epoch: int = 0, step_epoch: int = None, step_factor: float = 0.1,
                  power: float = 0.9):
    """return learning rate scheduler"""

    assert mode in ('constant', 'step', 'linear', 'poly', 'cosine'), \
        f"Unknown learning rate scheduler: {mode}"
    sched_kwargs = {
        'base_lr': base_lr,
        'target_lr': target_lr,
        'nepochs': nepochs,
        'iters_per_epoch': iters_per_epoch,
    }
    if mode == 'step':
        sched_kwargs['mode'] = 'step'
        sched_kwargs['step_epoch'] = step_epoch
        sched_kwargs['step_factor'] = step_factor
    elif mode == 'poly':
        sched_kwargs['mode'] = 'poly'
        sched_kwargs['power'] = power
    else:
        sched_kwargs['mode'] = mode
    return LRScheduler(**sched_kwargs)


class Fitter:
    """
    model training
    """

    def __init__(self, net, config, ctx_lst, val_per_epoch=True):
        self.net = net
        self.conf = config
        self.ctx = ctx_lst
        self.val = val_per_epoch
        # tools
        self.id = get_strftime()
        self.logger = get_logger(log_file=os.path.join(config.record_dir, f'{self.id}.log'))
        # data
        self.train_iter, self.num_train = _data_iter(config.data_name, config.bs_train, shuffle=True,
                                                     root=config.data_path, split='train', mode='train',
                                                     base_size=config.base, crop_size=config.crop)
        self.val_iter, self.num_val = _data_iter(config.data_name, config.bs_val, shuffle=False,
                                                 last_batch='keep', root=config.data_path, split='val',
                                                 base_size=config.base, crop_size=config.crop)
        # optimizer
        self.criterion = _criterion(config.aux, config.aux_weight)
        if config.optimizer == 'adam':
            self.trainer = _adam_trainer(self.net, config.lr, config.wd, config.adam_beta1,
                                         config.adam_beta2)
        else:
            self.trainer = _trainer(self.net, config.optimizer, config.lr, config.lr_scheduler,
                                    config.wd, config.momentum,
                                    target_lr=config.final_lr, nepochs=config.epochs,
                                    iters_per_epoch=len(self.train_iter), step_epoch=config.step_epoch,
                                    step_factor=config.step_factor, power=config.poly_pwr)
        # metric
        self.metric = SegmentationMetric(nclass=config.nclass)

    def log(self):
        """log hyper-params"""
        self.logger.info(f"Initialization complete for training: {self.id.replace('_', '-')}")
        self.logger.info(f"Training {self.conf.model_name} on {self.ctx}")
        self.logger.info(f"Training images: {self.num_train}, validation images: {self.num_val}")
        for k, v in self.conf.items():
            self.logger.info(f"{k}: {v}")
        self.logger.info('platform: ' + platform.system())
        self.logger.info('-------------------------')

    def fit(self):
        last_miou = .0  # record the best validation mIoU
        loss_step = 0  # step count
        for epoch in range(self.conf.epochs):
            train_loss = .0
            start = time.time()
            for i, (data, target) in enumerate(self.train_iter):
                gpu_datas = split_and_load(data, ctx_list=self.ctx)
                gpu_targets = split_and_load(target, ctx_list=self.ctx)
                with autograd.record():
                    loss_gpu = [self.criterion(*self.net(gpu_data), gpu_target)
                                for gpu_data, gpu_target in zip(gpu_datas, gpu_targets)]
                for loss in loss_gpu:
                    autograd.backward(loss)
                self.trainer.step(self.conf.bs_train)
                nd.waitall()
                loss_temp = .0
                for losses in loss_gpu:
                    loss_temp += losses.sum().asscalar()
                train_loss += (loss_temp / self.conf.bs_train)
                # log every n batch
                # add loss to draw curve, train_loss <class numpy.float64>
                interval = 5 if loss_step < 5000 else 50
                if (i % interval == 0) or (i + 1 == len(self.train_iter)):
                    fitlog.add_loss(name='loss', value=round(train_loss / (i + 1), 5), step=loss_step)
                    loss_step += 1
                    self.logger.info(
                        "Epoch %d, batch %d, training loss %.5f." % (epoch, i, train_loss / (i + 1)))
            # log each epoch
            self.logger.info(
                ">>>>>> Epoch %d complete, time cost: %.1f sec. <<<<<<" % (epoch, time.time() - start))
            # validation each epoch
            if self.val:
                pixel_acc, mean_iou = self._validation()
                self.logger.info(
                    "Epoch %d validation, PixelAccuracy: %.4f, mIoU: %.4f." % (epoch, pixel_acc, mean_iou))
                fitlog.add_metric(value=mean_iou, step=epoch, name='mIoU')
                fitlog.add_metric(value=pixel_acc, step=epoch, name='PA')
                if mean_iou > last_miou:
                    f_name = self._save_model(tag='best')
                    self.logger.info("Epoch %d mIoU: %.4f > %.4f(previous), save model: %s"
                                     % (epoch, mean_iou, last_miou, f_name))
                    last_miou = mean_iou

        # save the final-epoch params
        f_name = self._save_model(tag='last')
        self.logger.info(">>>>>> Training complete, save model: %s. <<<<<<" % f_name)
        # record
        fitlog.add_best_metric(value=round(last_miou, 4), name='mIoU')
        fitlog.add_other(value=self.id, name='record_id')
        fitlog.add_other(value=self.num_train, name='train')
        fitlog.add_other(value=self.num_val, name='val')

    def _validation(self):
        """validation"""
        cudnn_auto_tune(False)
        tbar = tqdm(self.val_iter)
        for i, (data, label) in enumerate(tbar):
            gpu_datas = split_and_load(data=data, ctx_list=self.ctx, even_split=False)
            gpu_labels = split_and_load(data=label, ctx_list=self.ctx, even_split=False)
            for gpu_data, gpu_label in zip(gpu_datas, gpu_labels):
                self.metric.update(gpu_label, self.net.evaluate(gpu_data))
            tbar.set_description('pixAcc: %.4f, mIoU: %.4f' % (self.metric.get()))
        pixel_acc, mean_iou = self.metric.get()
        self.metric.reset()
        cudnn_auto_tune(True)
        return pixel_acc, mean_iou

    def _save_model(self, tag):
        """ save model parameters. """
        filename = self.conf.model_name + '_' + self.conf.backbone + '_' + \
                   self.conf.data_name + '_' + self.id + '_' + tag + '.params'
        self.net.save_parameters(os.path.join(self.conf.model_dir, filename))
        if tag == 'last':
            self.net.export(os.path.join(self.conf.model_dir, filename.replace('.params', '')),
                            epoch=self.conf.epochs)
        return filename
