# coding=utf-8

from typing import Any
from mxnet.gluon import nn
from .activation import Activation

__all__ = ['conv_block', 'ConvBlock', 'InverseConvBlock',
           'DepthSepConvolution', 'SpaceSepConvolution']


def conv_block(inverse=False, **kwargs):
    """
    get Conv block.
    """
    if inverse:
        return InverseConvBlock(**kwargs)
    return ConvBlock(**kwargs)


class ConvBlock(nn.HybridBlock):
    """
    Convolution-BN-Activation block.
    """

    def __init__(self, channels, kernel_size, strides: Any = 1, padding: Any = 0, dilation=1,
                 groups=1, use_bias=False, in_channels=0, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, activation='relu'):
        super(ConvBlock, self).__init__()
        with self.name_scope():
            self.conv = nn.Conv2D(channels, kernel_size, strides, padding, dilation, groups,
                                  use_bias=use_bias, in_channels=in_channels)
            self.bn = norm_layer(**({} if norm_kwargs is None else norm_kwargs)) if norm_layer else None
            self.act = Activation(activation) if activation else None

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x


class InverseConvBlock(nn.HybridBlock):
    """
    BN-Activation-Convolution block.
    """

    def __init__(self, channels, kernel_size, strides: Any = 1, padding: Any = 0, dilation=1,
                 groups=1, use_bias=False, in_channels=0, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, activation='relu'):
        super(InverseConvBlock, self).__init__()
        with self.name_scope():
            self.bn = norm_layer(**({} if norm_kwargs is None else norm_kwargs)) if norm_layer else None
            self.act = Activation(activation) if activation else None
            self.conv = nn.Conv2D(channels, kernel_size, strides, padding, dilation, groups,
                                  use_bias=use_bias, in_channels=in_channels)

    def hybrid_forward(self, F, x, *args, **kwargs):
        if self.bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        x = self.conv(x)
        return x


class DepthSepConvolution(nn.HybridBlock):
    """
    Depth-wise separable convolution.
    Reference: Howard A G, Zhu M, Chen B, et al. Mobilenets: Efficient convolutional neural
        networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017.
    """

    def __init__(self, channels, in_channels, kernel_size, strides=1, padding=0, dilation=1,
                 use_bias=False, norm_layer=nn.BatchNorm, norm_kwargs=None, activation='relu'):
        super(DepthSepConvolution, self).__init__()
        with self.name_scope():
            self.depthwise = ConvBlock(in_channels, kernel_size, strides, padding, dilation, groups=in_channels,
                                       use_bias=use_bias, in_channels=in_channels, norm_layer=norm_layer,
                                       norm_kwargs=norm_kwargs, activation=activation)
            self.pointwise = ConvBlock(channels, 1, use_bias=use_bias, in_channels=in_channels,
                                       norm_layer=norm_layer, norm_kwargs=norm_kwargs, activation=activation)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SpaceSepConvolution(nn.HybridBlock):
    """
    Space separable Conv-BN-Activation block.
    """

    def __init__(self, channels, kernel_size, strides=1, padding=0, groups=1, use_bias=False,
                 in_channels=0, norm_layer=nn.BatchNorm, norm_kwargs=None, activation='relu'):
        super(SpaceSepConvolution, self).__init__()
        with self.name_scope():
            self.conv1 = ConvBlock(channels, kernel_size=(1, kernel_size), strides=(1, strides),
                                   padding=(0, padding), groups=groups, use_bias=use_bias, in_channels=in_channels,
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs, activation=activation)
            self.conv2 = ConvBlock(channels, kernel_size=(kernel_size, 1), strides=(strides, 1),
                                   padding=(padding, 0), groups=groups, use_bias=use_bias, in_channels=in_channels,
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs, activation=activation)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
