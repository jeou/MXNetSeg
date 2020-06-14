# coding=utf-8

import os
import time
import platform

import mxnet as mx
from mxnet import nd
from mxnet.gluon.nn import BatchNorm
from gluoncv.nn import GroupNorm
from mxnet.gluon.contrib.nn.basic_layers import SyncBatchNorm


def cudnn_auto_tune(tune: bool = True):
    """
    Linux/Windows: MXNet cudnn auto-tune
    """
    tag = 1 if tune else 0
    if platform.system() == 'Linux':
        os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = str(tag)
    else:
        os.system(f"set MXNET_CUDNN_AUTOTUNE_DEFAULT={tag}")


def get_contexts(ctx_id):
    """
    Return all available GPUs, or [mx.cpu()] if there is no GPU

    :param ctx_id: list of context ids
    :return: list of MXNet contexts
    """
    ctx_list = []
    if ctx_id:
        try:
            for i in ctx_id:
                ctx = mx.gpu(i)
                _ = nd.array([0], ctx=ctx)
                ctx_list.append(ctx)
        except mx.base.MXNetError:
            raise RuntimeError(f"=> unknown gpu id: {ctx_id}")
    else:
        ctx_list = [mx.cpu()]
    return ctx_list


def get_strftime():
    """
    string format time
    """
    return time.strftime('%m_%d_%H_%M_%S', time.localtime())


def get_bn_layer(bn: str, ctx=(mx.cpu())):
    """
    get corresponding norm layer
    """
    if bn == 'bn':
        norm_layer = BatchNorm
        norm_kwargs = None
    elif bn == 'sbn':
        norm_layer = SyncBatchNorm
        norm_kwargs = {'num_devices': len(ctx)}
    elif bn == 'gn':
        norm_layer = GroupNorm
        norm_kwargs = {'ngroups': 32}
    else:
        raise NotImplementedError(f"Unknown batch normalization layer: {bn}")
    return norm_layer, norm_kwargs


def list_to_str(lst):
    """
    convert list/tuple to string split by space
    """
    result = " ".join(str(i) for i in lst)
    return result

