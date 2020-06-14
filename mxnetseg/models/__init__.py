# coding=utf-8

import mxnet as mx
from ..tools import get_bn_layer

from .base import *
from .fcn import *
from .pspnet import *
from .swiftnet import *
from .deeplabv3 import *
from .deeplabv3plus import *
from .denseaspp import *
from .ladder import *
from .seenet import *
from .bisenet import *
from .att2scale import *

from .canet import *
from .ssnet import *
from .next import *

_seg_models = {
    'fcn': (get_fcn, fcn_conf()),
    'pspnet': (get_pspnet, pspnet_conf()),
    'swiftnet': (get_swiftnet, swiftnet_conf()),
    'deeplabv3': (get_deeplabv3, deeplabv3_conf()),
    'deeplabv3plus': (get_deeplabv3plus, deeplabv3plus_conf()),
    'denseaspp': (get_denseaspp, denseaspp_conf()),
    'ladder': (get_ladder, ladder_conf()),
    'seenet': (get_seenet, seenet_conf()),
    'bisenet': (get_bisenet, bisenet_conf()),
    'att2scale': (get_att2scale, att2scale_conf()),
    'canet': (get_canet, canet_conf()),
    'ssnet': (get_ssnet, ssnet_conf()),
    'next': (get_next, next_conf()),
}


def get_model_by_name(name, ctx=(mx.cpu()), model_kwargs=None):
    """get model by name"""
    get_model, conf = _seg_models[name.lower()]
    if model_kwargs is None:
        # training
        norm_layer, norm_kwargs = get_bn_layer(conf.norm, ctx)
        model_kwargs = {'nclass': conf.nclass,
                        'backbone': conf.backbone,
                        'aux': conf.aux,
                        'base_size': conf.base,
                        'crop_size': conf.crop,
                        'pretrained_base': conf.pretrained_base,
                        'norm_layer': norm_layer,
                        'norm_kwargs': norm_kwargs,
                        'dilate': conf.dilate if 'dilate' in conf.keys() else False, }
        model = get_model(**model_kwargs)
        if conf.resume:
            model.load_parameters(conf.resume)
        else:
            _init_model(model, conf.pretrained_base, conf.lr_multiplier)
        model.collect_params().reset_ctx(ctx=ctx)
        return model, conf
    else:
        # evaluation
        model = get_model(**model_kwargs)
        return model


def _init_model(model, pretrained_base, lr_mult=10):
    if not pretrained_base:
        model.initialize()
    else:
        model.head.initialize()
        model.head.collect_params().setattr('lr_mult', lr_mult)
        if model.aux:
            model.auxlayer.initialize()
            model.auxlayer.collect_params().setattr('lr_mult', lr_mult)
