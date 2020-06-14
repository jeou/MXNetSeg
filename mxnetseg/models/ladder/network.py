# coding=utf-8

from mxnet.gluon import nn
from ..base import SegBaseDenseNet
from mxnetseg.modules import FCNHead, ConvBlock

__all__ = ['get_ladder', 'LadderDenseNet']


class LadderDenseNet(SegBaseDenseNet):
    """
    ladder-style DenseNet.
    Reference: Krešo, I., Šegvić, S., & Krapac, J. (2018). Ladder-Style DenseNets for
        Semantic Segmentation of Large Natural Images.
        In IEEE International Conference on Computer Vision (pp. 238–245).
        https://doi.org/10.1109/ICCVW.2017.37
    """

    def __init__(self, nclass, backbone='densnet169', aux=True, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=False,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(LadderDenseNet, self).__init__(nclass, aux, backbone, height, width, base_size,
                                             crop_size, pretrained_base, norm_layer=norm_layer,
                                             norm_kwargs=norm_kwargs)
        decoder_capacity = 128
        with self.name_scope():
            self.head = _LadderHead(nclass, decoder_capacity, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                    input_height=self._up_kwargs['height'],
                                    input_width=self._up_kwargs['width'])
            if self.aux:
                self.auxlayer = FCNHead(nclass=nclass, in_channels=decoder_capacity, norm_layer=norm_layer,
                                        norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c1, c2, c3, c4 = self.base_forward(x)
        outputs = []
        x, c4 = self.head(c4, c3, c2, c1)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c4)
            auxout = F.contrib.BilinearResize2D(auxout, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)

    def predict(self, x):
        h, w = x.shape[2:]
        self._up_kwargs['height'] = h
        self._up_kwargs['width'] = w
        self.head.fusion_c3.up_kwargs['height'] = h // 16
        self.head.fusion_c3.up_kwargs['width'] = w // 16
        self.head.fusion_c2.up_kwargs['height'] = h // 8
        self.head.fusion_c2.up_kwargs['width'] = w // 8
        self.head.fusion_c1.up_kwargs['height'] = h // 4
        self.head.fusion_c1.up_kwargs['width'] = w // 4
        return self.forward(x)[0]


class _LadderHead(nn.HybridBlock):
    """decoder for LadderResNet"""

    def __init__(self, nclass, decoder_capacity, input_height, input_width,
                 norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_LadderHead, self).__init__()
        with self.name_scope():
            self.conv_c4 = ConvBlock(decoder_capacity, 1, norm_layer=norm_layer,
                                     norm_kwargs=norm_kwargs, activation='relu')
            self.fusion_c3 = _LadderFusion(decoder_capacity, input_height // 16, input_width // 16,
                                           norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.fusion_c2 = _LadderFusion(decoder_capacity, input_height // 8, input_width // 8,
                                           norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.fusion_c1 = _LadderFusion(decoder_capacity, input_height // 4, input_width // 4,
                                           norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.seg_head = FCNHead(nclass, decoder_capacity, norm_layer, norm_kwargs, activation='relu')

    def hybrid_forward(self, F, x, *args, **kwargs):
        c4, c3, c2, c1 = x, args[0], args[1], args[2]
        c4 = self.conv_c4(c4)
        out = self.fusion_c3(c4, c3)
        out = self.fusion_c2(out, c2)
        out = self.fusion_c1(out, c1)
        out = self.seg_head(out)
        return out, c4


class _LadderFusion(nn.HybridBlock):
    """fusion module of U-shape structure"""

    def __init__(self, capacity, height, width, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_LadderFusion, self).__init__()
        with self.name_scope():
            self.conv1x1 = ConvBlock(capacity, 1, norm_layer=norm_layer,
                                     norm_kwargs=norm_kwargs, activation='relu')
            self.conv3x3 = ConvBlock(capacity, 3, 1, 1, norm_layer=norm_layer,
                                     norm_kwargs=norm_kwargs, activation='relu')

        self.up_kwargs = {'height': height, 'width': width}

    def hybrid_forward(self, F, x, *args, **kwargs):
        high, low = x, args[0]
        high = F.contrib.BilinearResize2D(high, **self.up_kwargs)
        low = self.conv1x1(low)
        out = high + low
        return self.conv3x3(out)


def get_ladder(**kwargs):
    return LadderDenseNet(**kwargs)
