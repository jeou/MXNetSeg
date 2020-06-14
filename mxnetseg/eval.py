# coding=utf-8

import os
import sys
import argparse

# runtime environment
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from mxnetseg.models import get_model_by_name
from mxnetseg.data import get_dataset_info
from mxnetseg.tools import get_logger, get_contexts, get_bn_layer, weight_path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--eval', type=str, default='metrics',
                        choices=('metrics', 'summary', 'speed', 'visual'),
                        help="evaluation mode")

    # model settings for all evaluation modes
    parser.add_argument('--model', type=str, default='fcn', help="model name")
    parser.add_argument('--backbone', type=str, default='resnet18', help="backbone network")
    parser.add_argument('--norm', type=str, default='sbn', help="norm layer, bn = sbn when evaluation")
    parser.add_argument('--dilate', action='store_true', default=False, help="dilated backbone")
    parser.add_argument('--aux', action='store_true', default=False, help="auxiliary loss")
    parser.add_argument('--resume', type=str,
                        default='fcn_resnet18_Cityscapes_06_11_10_05_29_best.params',
                        help='checkpoint')

    # devices for all evaluation modes
    parser.add_argument('--ctx', type=int, nargs='+', default=(0,),
                        help="ids of GPUs or leave None to use CPU")

    # data settings for all evaluation modes
    parser.add_argument('--data', type=str, default='Cityscapes', help="dataset name")
    parser.add_argument('--crop', type=int, default=768, help="crop size")
    parser.add_argument('--base', type=int, default=2048, help="random scale base size")
    parser.add_argument('--mode', type=str, default='val',
                        choices=('val', 'test', 'testval'),
                        help="evaluation/prediction on val/test set")
    parser.add_argument('--ms', action='store_true', default=True,
                        help="enable multi-scale and flip skills")
    parser.add_argument('--results', type=str,
                        # default='C:\\Users\\BedDong\\Desktop\\results\\VOC2012\\Segmentation\\comp6_test_cls',
                        default='C:\\Users\\BedDong\\Desktop\\results\\cityscapes',
                        help='path to save predictions for windows')

    # speed/summary settings
    parser.add_argument('--size', type=int, nargs='+', default=(1024, 2048),
                        help="image shape: height x width")
    parser.add_argument('--iterations', type=int, default=1000,
                        help="iterations for speed test")
    parser.add_argument('--bs', type=int, default=1, help="batch size should be 1")

    # visualization settings 
    parser.add_argument('--sample', type=str, default='2007_000663',
                        help="sample image name")
    parser.add_argument('--shape', type=int, nargs='+', default=(480, 480),
                        help="image shape: width x height")
    parser.add_argument('--tag', type=str, default='affinity',
                        help="sample sub-folder in the demo folder")

    arg = parser.parse_args()
    return arg


def _validate_checkpoint(model_name: str, checkpoint: str):
    """get model params"""
    checkpoint = os.path.join(weight_path(model_name), checkpoint)
    if not os.path.isfile(checkpoint):
        raise RuntimeError(f"No model params found at {checkpoint}")
    return checkpoint


if __name__ == '__main__':
    """args"""
    args = parse_args()
    logger = get_logger()

    """context"""
    ctx = get_contexts(args.ctx)

    """ load model """
    logger.info(f"Loading model [{args.model}] for [{args.eval}] on {args.data} ...")
    data_path, nclass = get_dataset_info(args.data)
    norm_layer, norm_kwargs = get_bn_layer(args.norm, ctx)
    model_kwargs = {'nclass': nclass,
                    'backbone': args.backbone,
                    'aux': args.aux,
                    'base_size': args.base,
                    'crop_size': args.crop,
                    'pretrained_base': False,
                    'norm_layer': norm_layer,
                    'norm_kwargs': norm_kwargs,
                    'dilate': args.dilate, }
    model = get_model_by_name(args.model, ctx, model_kwargs)

    resume = _validate_checkpoint(args.model, args.resume)
    model.load_parameters(resume, ctx=ctx)

    """evaluation on: metrics/summary/speed/visual"""
    if args.eval == 'metrics':
        from mxnetseg.furnace import Evaluator

        evaluator = Evaluator(model, args.data, mode=args.mode, ms=args.ms, results_dir=args.results,
                              data_path=data_path, nclass=nclass, ctx=ctx)
        evaluator.eval(logger)
        exit()

    if args.eval in ('summary', 'speed'):
        from mxnetseg.furnace import Speedometer

        speedometer = Speedometer(model, args.size, ctx[0], args.bs)
        if args.eval == 'speed':
            logger.info("Total time: %.2f s    Avg time per image: %.2f ms    FPS: %.1f" % (
                speedometer.speed(iterations=args.iterations)))
        else:
            speedometer.summary()
        exit()

    if args.eval == 'visual':
        # from mxnetseg.flash import Alchemy
        # from mxnetseg.tools import get_demo_image_path, load_image, apply_transform, overlap_heat_map, plot_heat_map
        #
        # sample_pth = get_demo_image_path(args.sample, args.tag)
        # sample = load_image(sample_pth, args.shape)
        # sample = apply_transform(sample)
        # alchemy = Alchemy(sample, model, ctx=ctx[0])
        # alchemy.forward()
        #
        # pixel_similarity = alchemy.pixels_similarity()
        # plot_heat_map(pixel_similarity, sns=True)
        #
        # alchemy.down_sample_prediction()
        # mask_similarity = alchemy.mask_similarity()
        # plot_heat_map(mask_similarity, sns=True)

        pass
