# coding=utf-8

from PIL import Image
from gluoncv.utils.viz import get_color_pallete

__all__ = ['get_color_palette', 'city_train2label']


def city_train2label(npimg):
    """convert train ID to label ID for cityscapes"""
    import numpy as np
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                     23, 24, 25, 26, 27, 28, 31, 32, 33]
    pred = np.array(npimg, np.uint8)
    label = np.zeros(pred.shape)
    ids = np.unique(pred)
    for i in ids:
        label[np.where(pred == i)] = valid_classes[i]
    out_img = Image.fromarray(label.astype('uint8'))
    return out_img


def get_color_palette(npimg, dataset: str = 'pascal_voc'):
    """
    Visualize image and return PIL.Image with color palette.

    :param npimg: Single channel numpy image with shape `H, W, 1`
    :param dataset: dataset name
    """

    if dataset in ('pascal_voc', 'ade20k', 'citys', 'mhpv1'):
        return get_color_pallete(npimg, dataset=dataset)

    elif dataset == 'camvid':
        npimg[npimg == -1] = 11
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(cam_palette)
        return out_img
    elif dataset == 'kittizhang':
        npimg = npimg + 1
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(kittizhang_palette)
        return out_img
    elif dataset == 'kittixu':
        npimg = npimg + 1
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(kittixu_palette)
        return out_img
    elif dataset == 'kittiros':
        npimg[npimg == -1] = 11
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(kittiros_palette)
        return out_img
    elif dataset == 'gatech':
        npimg = npimg + 1
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(gatech_palette)
        return out_img
    else:
        raise RuntimeError("Un-defined palette for data {}".format(dataset))


cam_palette = [
    128, 128, 128,  # sky
    128, 0, 0,  # building
    192, 192, 128,  # column_pole
    128, 64, 128,  # road
    0, 0, 192,  # sidewalk
    128, 128, 0,  # tree
    192, 128, 128,  # SignSymbol
    64, 64, 128,  # fence
    64, 0, 128,  # car
    64, 64, 0,  # pedestrian
    0, 128, 192,  # bicyclist
    0, 0, 0  # void
]

kittizhang_palette = [
    255, 255, 255,  # void
    153, 0, 0,  # building
    0, 51, 102,  # sky
    160, 160, 160,  # road
    0, 102, 0,  # vegetation
    255, 228, 196,  # sidewalk
    255, 200, 50,  # car
    255, 153, 255,  # pedestrian
    204, 153, 255,  # cyclist
    130, 255, 255,  # signage
    193, 120, 87  # fence
]

kittiros_palette = [
    128, 128, 128,  # sky
    128, 0, 0,  # building
    128, 64, 128,  # road
    0, 0, 192,  # sidewalk
    64, 64, 128,  # fence
    128, 128, 0,  # vegetation
    192, 192, 128,  # pole
    64, 0, 128,  # car
    192, 128, 128,  # sign
    64, 64, 0,  # pedestrian
    0, 128, 192,  # cyclist
    0, 0, 0]  # void

kittixu_palette = [
    0, 51, 153,  # void
    0, 51, 153,  # sky
    255, 192, 0,  # ground
    255, 192, 0,
    255, 192, 0,
    255, 192, 0,
    255, 192, 0,
    0, 255, 0,  # vegetation
    0, 255, 0,
    0, 255, 0,
    111, 159, 255,  # infrastructure
    111, 159, 255,
    111, 159, 255,
    111, 159, 255,
    111, 159, 255,
    111, 159, 255,
    111, 159, 255,
    255, 0, 0,  # movable obstacle
    255, 0, 0,
    255, 0, 0,
    255, 0, 0,
    255, 0, 0,
    255, 0, 0,
    255, 0, 0,
    255, 0, 0,
    255, 0, 0,
    255, 0, 0,
    255, 0, 0,
    255, 0, 0,
]

# self-defined GATECH palette
gatech_palette = [
    128, 128, 128,  # sky
    128, 64, 128,  # ground
    128, 0, 0,  # solid buildings etc
    192, 128, 128,  # porous
    64, 64, 0,  # humans
    64, 0, 128,  # cars
    0, 128, 192,  # vertical Mix
    0, 0, 192,  # main Mix
]
