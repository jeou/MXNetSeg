# coding=utf-8

from mxnet.gluon import nn
from mxnet.context import cpu
from mxnetseg.modules import ConvBlock, Activation

__all__ = ['Xception', 'xception39']


class SeparableConvBlock(nn.HybridBlock):
    def __init__(self, in_channels, channels, kernel_size, strides=1, padding=0, dilation=1,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, activation='relu'):
        super(SeparableConvBlock, self).__init__()
        self.conv = nn.Conv2D(in_channels, kernel_size, strides, padding, dilation,
                              groups=in_channels, use_bias=False, in_channels=in_channels)
        self.pointwise = ConvBlock(channels, 1, use_bias=False, in_channels=in_channels,
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs, activation=activation)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv(x)
        x = self.pointwise(x)
        return x


class Block(nn.HybridBlock):
    expansion = 4

    def __init__(self, in_channels, mid_channels, strides, dilation=1, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, activation='relu'):
        super(Block, self).__init__()
        if strides > 1:
            self.down = ConvBlock(mid_channels * self.expansion, 1, strides=strides, use_bias=False,
                                  in_channels=in_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                  activation=None)
        else:
            self.down = None

        self.residual = nn.HybridSequential()
        self.residual.add(
            SeparableConvBlock(in_channels, mid_channels, 3, strides, dilation, dilation,
                               norm_layer=norm_layer, norm_kwargs=norm_kwargs, activation=activation),
            SeparableConvBlock(mid_channels, mid_channels, 3, 1, 1,
                               norm_layer=norm_layer, norm_kwargs=norm_kwargs, activation=activation),
            SeparableConvBlock(mid_channels, mid_channels * self.expansion, 3, 1, 1,
                               norm_layer=norm_layer, norm_kwargs=norm_kwargs, activation=None))
        self.act = Activation(activation)

    def hybrid_forward(self, F, x, *args, **kwargs):
        short_cut = self.down(x) if self.down else x
        residual = self.residual(x)
        out = short_cut + residual
        out = self.act(out)
        return out


class Xception(nn.HybridBlock):
    def __init__(self, block, layers, channels, classes=1000, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, **kwargs):
        super(Xception, self).__init__()
        self.in_channels = 8
        self.norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        with self.name_scope():
            self.conv1 = ConvBlock(self.in_channels, 3, 2, 1, use_bias=False, norm_layer=norm_layer,
                                   norm_kwargs=norm_kwargs, activation='relu')
            self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            self.layer1 = self._make_layer(block, norm_layer, layers[0], channels[0], strides=2)
            self.layer2 = self._make_layer(block, norm_layer, layers[1], channels[1], strides=2)
            self.layer3 = self._make_layer(block, norm_layer, layers[2], channels[2], strides=2)
            self.avgpool = nn.GlobalAvgPool2D()
            self.flat = nn.Flatten()
            self.fc = nn.Dense(in_units=self.in_channels, units=classes)

    def _make_layer(self, block, norm_layer, blocks, mid_channels, strides):
        layers = nn.HybridSequential()
        layers.add(block(self.in_channels, mid_channels, strides=strides, norm_layer=norm_layer,
                         norm_kwargs=self.norm_kwargs))
        self.in_channels = mid_channels * block.expansion
        for i in range(1, blocks):
            layers.add(block(self.in_channels, mid_channels, strides=1, norm_layer=norm_layer,
                             norm_kwargs=self.norm_kwargs))
        return layers

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.flat(x)
        x = self.fc(x)
        return x


def xception39(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """get Xception39 model"""
    model = Xception(Block, [4, 8, 4], [16, 32, 64], **kwargs)
    if pretrained:
        import os

        params = os.path.join(root, 'xception39.params')
        if os.path.exists(params):
            model.load_parameters(params, ctx=ctx)
        else:
            raise FileNotFoundError(f"Unavailable params file: {params}")
    return model
