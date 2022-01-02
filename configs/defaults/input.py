from yacs.config import CfgNode as CN

INPUT = CN()
INPUT.SIZE = (224, 224)
INPUT.INTERPOLATION = "bilinear"             # Mode of interpolation in resize functions
INPUT.TRANSFORMS = ()                        #  For available choices please refer to transforms.py
INPUT.NO_TRANSFORM = False                   # If True, tfm_train and tfm_test will be None
INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]     # Default mean and std come from ImageNet
INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
INPUT.CROP_PADDING = 4                       # Padding for random crop

"""Cutout"""
INPUT.CUTOUT_N = 1
INPUT.CUTOUT_LEN = 16

"""Gaussian noise"""
INPUT.GN_MEAN = 0.0
INPUT.GN_STD = 0.15

""" RandomAugment """
INPUT.RANDAUGMENT_N = 2
INPUT.RANDAUGMENT_M = 10

""" ColorJitter (brightness, contrast, saturation, hue)"""
INPUT.COLORJITTER_B = 0.4
INPUT.COLORJITTER_C = 0.4
INPUT.COLORJITTER_S = 0.4
INPUT.COLORJITTER_H = 0.1

INPUT.RGS_P = 0.2    # Random gray scale's probability

""" Gaussian blur """
INPUT.GB_P = 0.5  # propability of applying this operation
INPUT.GB_K = 21  # kernel size (should be an odd number)
