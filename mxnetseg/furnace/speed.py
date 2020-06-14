# coding=utf-8

import time
from mxnet import nd, autograd


def _sample(shape, context, bs) -> nd.NDArray:
    """ get a random sample """
    if isinstance(shape, (list, tuple)):
        h = shape[0]
        w = shape[1]
    else:
        h = shape
        w = shape
    sample = nd.random.uniform(shape=(bs, 3, h, w), ctx=context)
    return sample


class Speedometer:
    """model speed test"""

    def __init__(self, net, shape, ctx, bs=1):
        self.net = net
        self.sample = _sample(shape, ctx, bs)
        self.ctx = ctx
        self.bs = bs
        self.net.initialize(ctx=self.ctx, force_reinit=True)

    def speed(self, iterations=1000, warm_up=500):
        """speed test with hybridized HybridBlock"""
        self.net.hybridize(static_alloc=True)

        # warm-up to obtain stable speed
        print("Warm-up for %d forward passes..." % warm_up)
        for _ in range(warm_up):
            with autograd.record(False):
                self.net.predict(self.sample)
            nd.waitall()

        # speed test
        print("Speed test for %d forward passes..." % iterations)
        t_start = time.time()
        for _ in range(iterations):
            with autograd.record(False):
                self.net.predict(self.sample)
        nd.waitall()
        time_cost = time.time() - t_start
        return time_cost, (time_cost / iterations) * 1000 / self.bs, iterations * self.bs / time_cost

    def summary(self):
        """model summary"""
        self.net.summary(self.sample)
