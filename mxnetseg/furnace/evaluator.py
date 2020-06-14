# coding=utf-8

import os
import platform
from tqdm import tqdm

from mxnet import nd
from mxnet.gluon.data import DataLoader
from gluoncv.utils.metrics import SegmentationMetric
from gluoncv.model_zoo.segbase import MultiEvalModel

from mxnetseg.data import segmentation_dataset
from mxnetseg.tools import image_transform, root_path, cudnn_auto_tune

color_palette = {'voc2012': 'pascal_voc',
                 'ade20k': 'ade20k',
                 'cityscapes': 'citys',
                 'camvid': 'camvid',
                 'camvidfull': 'camvid',
                 'kittizhang': 'kittizhang',
                 'kittixu': 'kittixu',
                 'kittiros': 'kittiros', }


def _data_set(data_name, mode, data_path):
    """return dataset"""
    trans = image_transform()
    if mode == 'test':
        dataset = segmentation_dataset(data_name, root=data_path, split='test', mode='test', transform=trans)
    elif mode == 'testval':
        dataset = segmentation_dataset(data_name, root=data_path, split='val', mode='test', transform=trans)
    elif mode == 'val':
        dataset = segmentation_dataset(data_name, root=data_path, split='val', mode='testval', transform=trans)
    else:
        raise RuntimeError(f"Unknown mode: {mode}")
    return dataset


def _scales(ms, data_name):
    """return evaluation scale"""
    if ms:
        if data_name.lower() in ('cityscapes', 'camvid'):
            scales = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25)
        elif data_name.lower() in ('gatech',):
            scales = (0.5, 0.8, 1.0, 1.2, 1.4)
        else:
            scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
    else:
        scales = (1.0,)
    return scales


def _results_dir(results_dir):
    """results directory to save predictions"""
    if platform.system() == 'Linux':
        results_dir = os.path.join(root_path(), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir


class Evaluator:
    """model evaluation"""

    def __init__(self, net, data_name, mode, data_path, nclass, ms, ctx, results_dir):
        """
        initialize

        :param net: nn.HybridBlock model
        :param data_name: dataset name
        :param mode: dataset loading mode
        :param data_path:  dataset path
        :param nclass: number of classes
        :param ms: whether to use multi-scale and flipping skills
        :param ctx: contexts list
        """
        self.ctx = ctx[0]
        self.net = net
        self.net.hybridize()
        self.data_name = data_name
        self.mode = mode
        self.nclass = nclass
        self.results = results_dir
        # dataset
        self.dataset = _data_set(data_name, mode, data_path)
        # evaluator
        self.scales = _scales(ms, data_name)
        self.evaluator = MultiEvalModel(net, nclass, ctx, flip=ms, scales=self.scales)

    def eval(self, logger):
        """do evaluation/prediction"""

        logger.info("%s with scales: %s" % (self.mode, self.scales))
        cudnn_auto_tune(False)
        if 'test' in self.mode:
            self._predict()
        else:
            pa, iou = self._eval()
            logger.info("PA: %.4f    mIoU: %.4f" % (pa, iou))

    def _eval(self):
        """evaluation"""
        metric = SegmentationMetric(self.nclass)
        val_iter = DataLoader(self.dataset, batch_size=1, last_batch='keep')
        bar = tqdm(val_iter)
        for _, (img, label) in enumerate(bar):
            pred = self.evaluator.parallel_forward(img)[0]
            metric.update(label, pred)
            bar.set_description("PixelAccuracy: %.4f    mIoU: %.4f" % metric.get())
        return metric.get()

    def _predict(self):
        """prediction"""
        assert self.data_name.lower() in color_palette.keys()
        results_dir = _results_dir(self.results)
        bar = tqdm(self.dataset)

        # prediction without aux_out
        for _, (img, dst) in enumerate(bar):
            img = img.expand_dims(0)
            output = self.evaluator.parallel_forward(img)[0]
            mask = self._mask(output)
            # save predictions
            save_name = dst.split('.')[0] + '.png'
            save_path = os.path.join(results_dir, save_name)
            dir_path, _ = os.path.split(save_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            mask.save(save_path)

    def _mask(self, output: nd.NDArray, test_citys: bool = True):
        """get mask by output"""
        predict = nd.squeeze(nd.argmax(output, 1)).asnumpy()
        if test_citys:
            from mxnetseg.tools import city_train2label
            mask = city_train2label(predict)
        else:
            from mxnetseg.tools import get_color_palette
            mask = get_color_palette(predict, color_palette[self.data_name.lower()])
        return mask
