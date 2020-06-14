# coding=utf-8

from easydict import EasyDict

from mxnetseg.data import get_dataset_info
from mxnetseg.tools import weight_path, record_path

C = EasyDict()
config = C

# model name
C.model_name = 'swiftnet'
C.model_dir = weight_path(C.model_name)
C.record_dir = record_path(C.model_name)

# dataset:
# COCO, VOC2012, VOCAug, SBD, PContext,
# Cityscapes, CamVid, CamVidFull, Stanford, GATECH, KITTIZhang, KITTIXu, KITTIRos
# NYU, SiftFlow, SUNRGBD, ADE20K
C.data_name = 'Cityscapes'
C.crop = 768
C.base = 2048
C.data_path, C.nclass = get_dataset_info(C.data_name)

# network
C.backbone = 'resnet18'
C.pretrained_base = True
C.norm = 'sbn'
C.aux = False
C.aux_weight = 0.4 if C.aux else None
C.lr_multiplier = 10

C.resume = None
# import os
# C.resume = os.path.join(C.model_dir, '*.params')

# optimizer: sgd, nag, adam
# lr_scheduler: constant, step, linear, poly, cosine
C.lr = 1e-3
C.wd = 1e-4
C.optimizer = 'sgd'

if any([C.optimizer == 'sgd', C.optimizer == 'nag']):

    C.lr_scheduler = 'poly'
    C.poly_pwr = 0.9 if C.lr_scheduler == 'poly' else None
    C.step_factor = 0.5 if C.lr_scheduler == 'step' else None
    C.step_epoch = (5,) if C.lr_scheduler == 'step' else None
    C.momentum = 0.9
    C.final_lr = 0

else:
    C.adam_beta1 = 0.9
    C.adam_beta2 = 0.999

# train
C.epochs = 200
C.bs_train = 8
C.bs_val = 16
