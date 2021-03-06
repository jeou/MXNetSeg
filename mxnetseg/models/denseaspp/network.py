# coding=utf-8

from mxnet.gluon import nn

from mxnetseg.modules import FCNHead, DenseASPPBlock
from ..base import SegBaseResNet

__all__ = ['get_denseaspp', 'DenseASPP']


class DenseASPP(SegBaseResNet):
    """
    Dilated ResNet50/101/152 based DenseASPP.
    Reference: Zhang, C., Li, Z., Yu, K., Yang, M., & Yang, K. (2018).
        DenseASPP for Semantic Segmentation in Street Scenes.
        In IEEE Conference on Computer Vision and Pattern Recognition (pp. 3684–3692).
        IEEE. https://doi.org/10.1109/cvpr.2018.00388
    """

    def __init__(self, nclass, backbone='resnet50', aux=True, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=False,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(DenseASPP, self).__init__(nclass, aux, backbone, height, width,
                                        base_size, crop_size, pretrained_base, dilate=True,
                                        norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _DenseASPPHead(nclass, 2048, norm_layer, norm_kwargs)
            if self.aux:
                self.auxlayer = FCNHead(nclass, 1024, norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        _, _, c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c4)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.contrib.BilinearResize2D(auxout, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)


class _DenseASPPHead(nn.HybridBlock):
    def __init__(self, nclass, in_channels, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_DenseASPPHead, self).__init__()
        with self.name_scope():
            self.dense_aspp = DenseASPPBlock(256, in_channels, norm_layer, norm_kwargs)
            self.head = FCNHead(nclass, 256, norm_layer, norm_kwargs, reduction=1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.dense_aspp(x)
        return self.head(x)


def get_denseaspp(**kwargs):
    return DenseASPP(**kwargs)
