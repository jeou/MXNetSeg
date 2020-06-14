# coding=utf-8

from gluoncv import loss

__all__ = ['MultiSoftmaxCrossEntropyLoss', 'PixelAffinityLoss']


class MultiSoftmaxCrossEntropyLoss(loss.SoftmaxCrossEntropyLoss):
    """
    SoftmaxCrossEntropyLoss with multiple aux weight.
    """

    def __init__(self, aux=False, aux_weight=None, ignore_label=-1, **kwargs):
        super(MultiSoftmaxCrossEntropyLoss, self).__init__(ignore_label=ignore_label, **kwargs)
        self.aux = aux
        if aux and aux_weight:
            if isinstance(aux_weight, float):
                self.aux_weight = (aux_weight,)
            elif isinstance(aux_weight, (list, tuple)):
                self.aux_weight = tuple(aux_weight)
            else:
                raise RuntimeError(
                    f"Unknown type of aux_weight: {type(aux_weight)}, "
                    f"which should be float or list of float")

    def _aux_forward(self, F, *inputs, **kwargs):
        assert len(self.aux_weight) == (len(inputs) - 2)
        label = inputs[-1]
        total_loss = super(MultiSoftmaxCrossEntropyLoss, self).hybrid_forward(F, inputs[0], label)
        pairs = self._get_pairs(inputs)
        for weight, pred in pairs:
            this_loss = super(MultiSoftmaxCrossEntropyLoss, self).hybrid_forward(F, pred, label)
            total_loss = total_loss + weight * this_loss

        return total_loss

    def _get_pairs(self, inputs):
        pairs = []
        for i in range(len(self.aux_weight)):
            pair = (self.aux_weight[i], inputs[i + 1])
            pairs.append(pair)
        return tuple(pairs)

    def hybrid_forward(self, F, *inputs, **kwargs):
        """ input order: pred1, pred2, ..., label """
        if self.aux:
            return self._aux_forward(F, *inputs, **kwargs)
        else:
            return super(MultiSoftmaxCrossEntropyLoss, self).hybrid_forward(F, *inputs, **kwargs)


class PixelAffinityLoss(loss.SoftmaxCrossEntropyLoss):
    """
    Auxiliary loss as pixel affinity loss.
    """

    def __init__(self, affinity=True, affinity_weight=0.2, ignore_label=-1, sub_sample=True,
                 height=None, width=None, affinity_size=36, l2loss=True, **kwargs):
        """
        Initialization. Sub-sample is adopted based on memory considerations.
        :param affinity: whether to adopt affinity loss besides the standard cross-entropy loss
        :param affinity_weight: affinity loss coefficient
        :param ignore_label: ignored label when compute loss
        :param sub_sample: whether to down-sample label
        :param height: sub-sample height
        :param width:  sub-sample width
        :param affinity_size: sub-sample size
        :param l2loss: Set true to use mean square error for affinity loss, binary cross-entropy
            loss otherwise. The label and prediction should have the same size.
        """
        super(PixelAffinityLoss, self).__init__(ignore_label=ignore_label, **kwargs)
        self.affinity = affinity
        self.weight = affinity_weight
        self.height = height if height else affinity_size
        self.width = width if width else affinity_size
        self.sub_sample = sub_sample
        if l2loss:
            from mxnet.gluon.loss import L2Loss
            self.affinity_loss = L2Loss()
        else:
            from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss
            self.affinity_loss = SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)

    def _transform_label(self, F, label):
        """
        Transform label to form pixel affinity label of shape Nx(HW)x(HW),
        where H=self.height and W=self.width if self.sub_sample=True.
        """
        if self.sub_sample:
            label = F.contrib.BilinearResize2D(F.expand_dims(label, axis=1),
                                               height=self.height, width=self.width)
            label = F.squeeze(label, axis=1)
        label = F.reshape(label, shape=(0, -1))
        return label[:, :, None] == label[:, None]

    def _affinity_forward(self, F, pred, affinity_pred, label, **kwargs):
        loss1 = super(PixelAffinityLoss, self).hybrid_forward(F, pred, label)
        affinity_map = self._transform_label(F, label)
        loss2 = self.affinity_loss(affinity_pred, affinity_map)
        return loss1 + self.weight * loss2

    def hybrid_forward(self, F, *inputs, **kwargs):
        if self.affinity:
            return self._affinity_forward(F, *inputs, **kwargs)
        else:
            return super(PixelAffinityLoss, self).hybrid_forward(F, *inputs, **kwargs)
