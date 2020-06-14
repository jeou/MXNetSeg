# coding=utf-8

from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import HybridConcurrent
from .naive import ConvBlock

__all__ = ['PyramidPooling', 'GlobalFlow', 'ASPP', 'DenseASPPBlock',
           'GCN', 'SeEModule', 'DepthWiseASPP']


class PyramidPooling(nn.HybridBlock):
    """
    Pyramid Pooling Module.
    Reference: Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. (2017).
        Pyramid Scene Parsing Network. In IEEE Conference on Computer Vision and
        Pattern Recognition (pp. 6230–6239). https://doi.org/10.1109/CVPR.2017.660
    """

    def __init__(self, in_channels, height=60, width=60, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, activation='relu', reduction=4):
        super(PyramidPooling, self).__init__()
        self.up_kwargs = {'height': height, 'width': width}
        with self.name_scope():
            self.conv1 = ConvBlock(in_channels // reduction, 1, in_channels=in_channels, norm_layer=norm_layer,
                                   norm_kwargs=norm_kwargs, activation=activation)
            self.conv2 = ConvBlock(in_channels // reduction, 1, in_channels=in_channels, norm_layer=norm_layer,
                                   norm_kwargs=norm_kwargs, activation=activation)
            self.conv3 = ConvBlock(in_channels // reduction, 1, in_channels=in_channels, norm_layer=norm_layer,
                                   norm_kwargs=norm_kwargs, activation=activation)
            self.conv4 = ConvBlock(in_channels // reduction, 1, in_channels=in_channels, norm_layer=norm_layer,
                                   norm_kwargs=norm_kwargs, activation=activation)

    def hybrid_forward(self, F, x, *args, **kwargs):
        feature1 = F.contrib.BilinearResize2D(self.conv1(F.contrib.AdaptiveAvgPooling2D(x, output_size=1)),
                                              **self.up_kwargs)
        feature2 = F.contrib.BilinearResize2D(self.conv2(F.contrib.AdaptiveAvgPooling2D(x, output_size=2)),
                                              **self.up_kwargs)
        feature3 = F.contrib.BilinearResize2D(self.conv3(F.contrib.AdaptiveAvgPooling2D(x, output_size=3)),
                                              **self.up_kwargs)
        feature4 = F.contrib.BilinearResize2D(self.conv4(F.contrib.AdaptiveAvgPooling2D(x, output_size=6)),
                                              **self.up_kwargs)
        return F.concat(x, feature1, feature2, feature3, feature4, dim=1)


class GlobalFlow(nn.HybridBlock):
    """
    Global Average Pooling to obtain global receptive field.
    """

    def __init__(self, channels, in_channels=0, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, height=60, width=60):
        super(GlobalFlow, self).__init__()
        self.up_kwargs = {'height': height, 'width': width}
        with self.name_scope():
            self.gap = nn.GlobalAvgPool2D()
            self.conv1x1 = ConvBlock(channels, 1, in_channels=in_channels, norm_layer=norm_layer,
                                     norm_kwargs=norm_kwargs, activation='relu')

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.gap(x)
        x = self.conv1x1(x)
        x = F.contrib.BilinearResize2D(x, **self.up_kwargs)
        return x


class GCN(nn.HybridBlock):
    """
    Global Convolution Network.
    Employing {1xr,rx1} and {rx1,1xr} to approximate the rxr convolution.
    Reference:
        Peng, Chao, et al. "Large Kernel Matters--Improve Semantic Segmentation by
        Global Convolutional Network." Proceedings of the IEEE conference on computer
        vision and pattern recognition. 2017.
    """

    def __init__(self, channels, k_size, in_channels=0):
        super(GCN, self).__init__()
        pad = int((k_size - 1) / 2)
        with self.name_scope():
            self.conv_l1 = nn.Conv2D(channels, kernel_size=(k_size, 1), padding=(pad, 0),
                                     in_channels=in_channels)
            self.conv_l2 = nn.Conv2D(channels, kernel_size=(1, k_size), padding=(0, pad),
                                     in_channels=channels)
            self.conv_r1 = nn.Conv2D(channels, kernel_size=(1, k_size), padding=(0, pad),
                                     in_channels=in_channels)
            self.conv_r2 = nn.Conv2D(channels, kernel_size=(k_size, 1), padding=(pad, 0),
                                     in_channels=channels)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        return x_l + x_r


class ASPP(nn.HybridBlock):
    """
    Atrous Spatial Pyramid Pooling w/ or w/o Object Context (oc).
    When set oc=True, self.pool has no height and weight params.
    Reference:
        Chen, L.-C., Papandreou, G., Schroff, F., & Adam, H. (2017).
        Rethinking Atrous Convolution for Semantic Image Segmentation.
        https://doi.org/10.1002/cncr.24278

        Yuan, Yuhui, and Jingdong Wang. "Ocnet: Object context network for scene parsing."
        arXiv preprint arXiv:1809.00916 (2018).
    """

    def __init__(self, channels=256, in_channels=0, norm_layer=nn.BatchNorm, norm_kwargs=None,
                 height=60, width=60, atrous_rates=(6, 12, 18), drop=.5, oc=False):
        super(ASPP, self).__init__()
        rate1, rate2, rate3 = tuple(atrous_rates)
        with self.name_scope():
            self.conv1x1 = ConvBlock(channels, 1, in_channels=in_channels, norm_layer=norm_layer,
                                     norm_kwargs=norm_kwargs, activation='relu')
            self.dilate1 = ConvBlock(channels, 3, padding=rate1, dilation=rate1, in_channels=in_channels,
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs, activation='relu')
            self.dilate2 = ConvBlock(channels, 3, padding=rate2, dilation=rate2, in_channels=in_channels,
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs, activation='relu')
            self.dilate3 = ConvBlock(channels, 3, padding=rate3, dilation=rate3, in_channels=in_channels,
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs, activation='relu')
            if oc:
                from gluoncv.model_zoo.action_recognition.non_local import NonLocal
                self.pool = nn.HybridSequential()
                self.pool.add(ConvBlock(channels, 3, 1, 1, in_channels=in_channels,
                                        norm_layer=norm_layer, norm_kwargs=norm_kwargs),
                              NonLocal(channels, 'gaussian', dim=2, norm_layer=norm_layer,
                                       norm_kwargs=norm_kwargs))
            else:
                self.pool = GlobalFlow(channels, in_channels, norm_layer, norm_kwargs, height, width)
            self.proj = ConvBlock(channels, 1, in_channels=5 * channels, norm_layer=norm_layer,
                                  norm_kwargs=norm_kwargs, activation='relu')
            self.drop = nn.Dropout(drop)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x1 = self.conv1x1(x)
        x6 = self.dilate1(x)
        x12 = self.dilate2(x)
        x18 = self.dilate3(x)
        xp = self.pool(x)
        out = F.concat(x1, x6, x12, x18, xp, dim=1)
        return self.drop(self.proj(out))


class DenseASPPBlock(nn.HybridBlock):
    """
    DenseASPP block.
        input channels = 2048 with dilated ResNet50/101/152
        output channels of 1x1 transition layer = 1024
        output channels of each dilate 3x3 Conv = 256
    Reference: Zhang, C., Li, Z., Yu, K., Yang, M., & Yang, K. (2018).
        DenseASPP for Semantic Segmentation in Street Scenes.
        In IEEE Conference on Computer Vision and Pattern Recognition (pp. 3684–3692).
        IEEE. https://doi.org/10.1109/cvpr.2018.00388
    """

    def __init__(self, channels=256, in_channels=2048, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, atrous_rates=(3, 6, 12, 18, 24)):
        super(DenseASPPBlock, self).__init__()
        rate1, rate2, rate3, rate4, rate5 = tuple(atrous_rates)
        out_dilate = in_channels // 8
        out_transition = in_channels // 2
        with self.name_scope():
            self.dilate1 = ConvBlock(out_dilate, 3, padding=rate1, dilation=rate1,
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.trans1 = ConvBlock(out_transition, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.dilate2 = ConvBlock(out_dilate, 3, padding=rate2, dilation=rate2,
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.trans2 = ConvBlock(out_transition, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.dilate3 = ConvBlock(out_dilate, 3, padding=rate3, dilation=rate3,
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.trans3 = ConvBlock(out_transition, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.dilate4 = ConvBlock(out_dilate, 3, padding=rate4, dilation=rate4,
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.trans4 = ConvBlock(out_transition, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.dilate5 = ConvBlock(out_dilate, 3, padding=rate5, dilation=rate5,
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.proj = ConvBlock(channels, 1, in_channels=3328, norm_layer=norm_layer,
                                  norm_kwargs=norm_kwargs, activation='relu')
            self.drop = nn.Dropout(0.5)

    def hybrid_forward(self, F, x, *args, **kwargs):
        out1 = self.dilate1(x)

        out2 = F.concat(x, out1, dim=1)
        out2 = self.trans1(out2)
        out2 = self.dilate2(out2)

        out3 = F.concat(x, out1, out2, dim=1)
        out3 = self.trans2(out3)
        out3 = self.dilate3(out3)

        out4 = F.concat(x, out1, out2, out3, dim=1)
        out4 = self.trans3(out4)
        out4 = self.dilate4(out4)

        out5 = F.concat(x, out1, out2, out3, out4, dim=1)
        out5 = self.trans4(out5)
        out5 = self.dilate5(out5)

        out = F.concat(x, out1, out2, out3, out4, out5, dim=1)
        out = self.proj(out)
        return self.drop(out)


class SeEModule(nn.HybridBlock):
    """
    Semantic Enhancement Module. i.e. modified ASPP block.
    Not employ Depth-wise separable convolution here since it saves little parameters.
    Reference:
        Pang, Yanwei, et al. "Towards bridging semantic gap to improve semantic segmentation."
        Proceedings of the IEEE International Conference on Computer Vision. 2019.
    """

    def __init__(self, channels=128, atrous_rates=(1, 2, 4, 8), norm_layer=nn.BatchNorm,
                 norm_kwargs=None, full_sample=False):
        super(SeEModule, self).__init__()
        rate1, rate2, rate3, rate4 = tuple(atrous_rates)
        with self.name_scope():
            self.branch1 = self._make_branch(rate1, channels, norm_layer, norm_kwargs, full_sample)
            self.branch2 = self._make_branch(rate2, channels, norm_layer, norm_kwargs, full_sample)
            self.branch3 = self._make_branch(rate3, channels, norm_layer, norm_kwargs, full_sample)
            self.branch4 = self._make_branch(rate4, channels, norm_layer, norm_kwargs, full_sample)
            self.proj = ConvBlock(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = F.concat(x, out1, out2, out3, out4, dim=1)
        return self.proj(out)

    @staticmethod
    def _make_branch(dilation, channels, norm_layer, norm_kwargs, full_sample):
        branch = nn.HybridSequential()
        branch.add(ConvBlock(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        if full_sample:
            branch.add(GCN(channels, 3, in_channels=channels))
        branch.add(ConvBlock(channels, 3, padding=dilation, dilation=dilation,
                             norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return branch


class DepthWiseASPP(nn.HybridBlock):
    """
    Depth-wise ASPP block.
    """

    def __init__(self, channels, atrous_rates, in_channels, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, activation='relu'):
        super(DepthWiseASPP, self).__init__()
        with self.name_scope():
            self.concurrent = HybridConcurrent(axis=1)
            self.concurrent.add(ConvBlock(channels, 1, in_channels=in_channels,
                                          norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                          activation=activation))
            for i in range(len(atrous_rates)):
                rate = atrous_rates[i]
                self.concurrent.add(ConvBlock(channels, 3, 1, padding=rate, dilation=rate,
                                              groups=in_channels, in_channels=in_channels,
                                              norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                              activation=activation))
            self.conv1x1 = ConvBlock(channels, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                     activation=activation)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.concurrent(x)
        x = self.conv1x1(x)
        return x
