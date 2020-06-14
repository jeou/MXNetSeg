# coding=utf-8

import os
import sys
import fitlog
import argparse
import platform

# runtime environment
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from mxnetseg.furnace import Fitter
from mxnetseg.models import get_model_by_name
from mxnetseg.tools import get_contexts

# fitlog.debug()
fitlog.commit(__file__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training semantic segmentation model",
        epilog="Example: python train.py --model fcn --ctx 0 1 2 3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', required=True, type=str, help="model name.")
    parser.add_argument('--ctx', type=int, nargs='+', default=(0, 1, 2, 3),
                        help="ids of GPUs or leave None to use CPU")
    parser.add_argument('--val', action='store_true', default=False,
                        help="whether to perform validation per epoch")
    arg = parser.parse_args()
    return arg


def record_hyper_params(hyper_dict: dict):
    for k, v in hyper_dict.items():
        if k not in ('model_dir', 'record_dir', 'data_path'):
            v = v if v is not None else '-'
            fitlog.add_hyper(value=str(v), name=str(k))
    if 'dilate' not in hyper_dict.keys():
        fitlog.add_hyper(value='-', name='dilate')
    fitlog.add_other(value=platform.system(), name='platform')


if __name__ == '__main__':

    # parse args
    args = parse_args()

    # get contexts
    ctx = get_contexts(args.ctx)

    # get initialized model by model name
    model, conf = get_model_by_name(args.model, ctx)
    model.hybridize()

    # record hyper-parameters
    record_hyper_params(conf)

    # training
    try:
        my_fitter = Fitter(model, conf, ctx, val_per_epoch=args.val)
        my_fitter.log()  # log to file
        my_fitter.fit()
        fitlog.finish()
    except Exception as e:
        fitlog.finish(status=1)
        print(e)
