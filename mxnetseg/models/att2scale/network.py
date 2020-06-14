# coding=utf-8

from mxnet.gluon import nn
from ..base import SegBaseResNet
from mxnetseg.modules import FCNHead, AuxHead, ConvBlock

__all__ = ['get_att2scale', 'AttentionToScale']


class AttentionToScale(SegBaseResNet):
    """
    ResNet based attention-to-scale model.
    Only support training with two scales of 1.0x and 0.5x.
    Reference: L. C. Chen, Y. Yang, J. Wang, W. Xu, and A. L. Yuille,
        “Attention to Scale: Scale-Aware Semantic Image Segmentation,”
        in IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 3640–3649.
    """

    def __init__(self, nclass, backbone='resnet50', aux=True, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=True, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, **kwargs):
        super(AttentionToScale, self).__init__(nclass, aux, backbone, height, width, base_size,
                                               crop_size, pretrained_base, dilate=True,
                                               norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _AttentionHead(nclass, num_scale=2, height=self._up_kwargs['height'] // 8,
                                       width=self._up_kwargs['width'] // 8,
                                       norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            if self.aux:
                self.auxlayer = AuxHead(nclass, self.base_channels[3], norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        _, _, _, c4 = self.base_forward(x)
        x_half = F.contrib.BilinearResize2D(x, height=self._up_kwargs['height'] // 2,
                                            width=self._up_kwargs['width'] // 2)
        _, _, _, c4h = self.base_forward(x_half)
        outputs = []
        x = self.head(c4, c4h)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c4)
            auxout = F.contrib.BilinearResize2D(auxout, **self._up_kwargs)
            outputs.append(auxout)
            auxouth = self.auxlayer(c4h)
            auxouth = F.contrib.BilinearResize2D(auxouth, **self._up_kwargs)
            outputs.append(auxouth)

        return tuple(outputs)

    def predict(self, x):
        height, width = x.shape[2:]
        self._up_kwargs['height'] = height
        self._up_kwargs['width'] = width
        self.head.up_kwargs['height'] = height // 8
        self.head.up_kwargs['width'] = width // 8
        return self.forward(x)[0]


class _AttentionHead(nn.HybridBlock):
    """
    Obtain soft attention weights and do weighted summation。
    Here we feed with features from the final stage of ResNet for attention.
    """

    def __init__(self, nclass, num_scale=2, height=60, width=60, norm_layer=nn.BatchNorm,
                 norm_kwargs=None):
        super(_AttentionHead, self).__init__()
        self.num_scale = num_scale
        self.up_kwargs = {'height': height, 'width': width}
        with self.name_scope():
            self.seg_head = FCNHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv3x3 = ConvBlock(512, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                     activation='relu')
            self.conv1x1 = nn.Conv2D(num_scale, 1, in_channels=512)

    def hybrid_forward(self, F, x, *args, **kwargs):
        # up-sample for same size, i.e. 8x because of dilation
        c4h = F.contrib.BilinearResize2D(args[0], **self.up_kwargs)
        # score map
        score_c4 = self.seg_head(x)
        score_c4h = self.seg_head(c4h)
        # obtain soft weights
        weights = F.concat(c4h, x, dim=1)
        weights = self.conv3x3(weights)
        weights = self.conv1x1(weights)
        weights = F.softmax(weights, axis=1)
        # reshape
        weights = F.expand_dims(weights, axis=2)  # (N, 2, 1, H, W)
        score_c4 = F.expand_dims(score_c4, axis=1)
        score_c4h = F.expand_dims(score_c4h, axis=1)
        out = F.concat(score_c4, score_c4h, dim=1)  # (N, 2, nclass, H, W)
        # weighted sum
        out = F.broadcast_mul(weights, out)
        out = F.sum(out, axis=1)
        return out


def get_att2scale(**kwargs):
    return AttentionToScale(**kwargs)
