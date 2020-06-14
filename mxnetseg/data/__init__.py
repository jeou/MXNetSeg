# coding=utf-8

import os
import numpy as np

from gluoncv.data import COCOSegmentation
from gluoncv.data import ADE20KSegmentation
from .voc import VOCSegmentation
from .sbd import SBDSegmentation
from .vocaug import VOCAugSegmentation
from .cityscapes import CitySegmentation
from .nyu import NYUSegmentation
from .pcontext import PContextSegmentation
from .sift import SiftFlowSegmentation
from .camvid import CamVidSegmentation
from .stanford import StanfordSegmentation
from .gatech import GATECHSegmentation
from .sunrgbd import SunRGBDSegmentation
from .mhp import MHPV1Segmentation
from .kitti import KITTIZhSementation, KITTIXuSementation, KITTIRosSementation

from mxnetseg.tools import dataset_path

__all__ = ['segmentation_dataset', 'get_dataset_info', 'verify_classes']

_data_sets = {
    'coco': (COCOSegmentation, 'COCO'),
    'voc2012': (VOCSegmentation, 'VOCdevkit'),
    'vocaug': (VOCAugSegmentation, 'VOCdevkit'),
    'sbd': (SBDSegmentation, 'VOCdevkit'),
    'pcontext': (PContextSegmentation, 'PContext'),
    'cityscapes': (CitySegmentation, 'Cityscapes'),
    'ade20k': (ADE20KSegmentation, 'ADE20K'),
    'nyu': (NYUSegmentation, 'NYU'),
    'siftflow': (SiftFlowSegmentation, 'SiftFlow'),
    'camvid': (CamVidSegmentation, 'CamVid'),
    'camvidfull': (CamVidSegmentation, 'CamVidFull'),
    'stanford': (StanfordSegmentation, 'Stanford10'),
    'kittizhang': (KITTIZhSementation, 'KITTI'),
    'kittixu': (KITTIXuSementation, 'KITTI'),
    'kittiros': (KITTIRosSementation, 'KITTI'),
    'gatech': (GATECHSegmentation, 'GATECH'),
    'sunrgbd': (SunRGBDSegmentation, 'SUNRGBD'),
    'mph': (MHPV1Segmentation, 'MHP'),
}


def segmentation_dataset(name: str, **kwargs):
    """
    Data loader for corresponding segmentation dataset

    :param name: dataset name
    :param kwargs: keywords depending on the API
    :return: corresponding data loader
    """
    dataset, _ = _data_sets[name.lower()]
    return dataset(**kwargs)


def get_dataset_info(name: str):
    """
    dataset basic information
    :param name: dataset name
    :return: dataset path and number of classes
    """
    dataset, dataset_dir = _data_sets[name.lower()]
    return os.path.join(dataset_path(), dataset_dir), dataset.NUM_CLASS


def verify_classes(name):
    """ to verify class labels """
    dataset, dataset_dir = _data_sets[name.lower()]
    data_dir = os.path.join(dataset_path(), dataset_dir)
    # train set
    train_set = dataset(data_dir, split='train')
    print("Train images: %d" % len(train_set))
    train_class_num = _verify_classes_assist([train_set])
    print("Train class label is: %s" % str(train_class_num))
    # val set
    val_set = dataset(data_dir, split='val')
    print("Val images: %d" % len(val_set))
    val_class_num = _verify_classes_assist([val_set])
    print("Val class label is: %s" % str(val_class_num))


def _verify_classes_assist(data_list):
    from tqdm import tqdm
    uniques = []
    for datas in data_list:
        tbar = tqdm(datas)
        for _, (_, mask) in enumerate(tbar):
            mask = mask.asnumpy()
            unique = np.unique(mask)
            for v in unique:
                if v not in uniques:
                    uniques.append(v)
    uniques.sort()
    return uniques
