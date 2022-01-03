from .ext_augment import (Random2DTranslation,InstanceNormalization,Cutout,GaussianNoise)
from PIL import Image
from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, CenterCrop, RandomCrop, ColorJitter,
    RandomApply, GaussianBlur, RandomGrayscale, RandomResizedCrop,
    RandomHorizontalFlip
)

from .autoaugment import SVHNPolicy, CIFAR10Policy, ImageNetPolicy
from .randaugment import RandAugment, RandAugment2, RandAugmentFixMatch

__all__ = [
    'build_transform',
    'to_tensor'
]

AVAI_CHOICES = [
    "random_flip",
    "random_resized_crop",
    "normalize",
    "instance_norm",
    "random_crop",
    "random_translation",
    "center_crop",  # This has become a default operation for test
    "cutout",
    "imagenet_policy",
    "cifar10_policy",
    "svhn_policy",
    "randaugment",
    "randaugment_fixmatch",
    "randaugment2",
    "gaussian_noise",
    "colorjitter",
    "randomgrayscale",
    "gaussian_blur",
]

INTERPOLATION_MODES = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "nearest": Image.NEAREST,
}


def build_transform(cfg, is_train=True, choices=None):
    """Build transformation function.

    Args:
        cfg (CfgNode): config.
        is_train (bool, optional): for training (True) or test (False).
            Default is True.
        choices (list, optional): list of strings which will overwrite
            cfg.INPUT.TRANSFORMS if given. Default is None.
    """
    if cfg.INPUT.NO_TRANSFORM:
        print("Note: no transform is applied!")
        return None

    if choices is None:
        choices = cfg.INPUT.TRANSFORMS

    for choice in choices:
        assert choice in AVAI_CHOICES

    target_size = f"{cfg.INPUT.SIZE[0]}x{cfg.INPUT.SIZE[1]}"

    normalize = Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

    if is_train:
        return _build_transform_train(cfg, choices, target_size, normalize)
    else:
        return _build_transform_test(cfg, choices, target_size, normalize)


def _build_transform_train(cfg, choices, target_size, normalize):
    print("Building transform_train")
    tfm_train = []

    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]

    # Make sure the image size matches the target size
    conditions = []
    conditions += ["random_crop" not in choices]
    conditions += ["random_resized_crop" not in choices]
    if all(conditions):
        print(f"+ resize to {target_size}")
        tfm_train += [Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]

    if "random_translation" in choices:
        print("+ random translation")
        tfm_train += [
            Random2DTranslation(cfg.INPUT.SIZE[0], cfg.INPUT.SIZE[1])
        ]

    if "random_crop" in choices:
        crop_padding = cfg.INPUT.CROP_PADDING
        print("+ random crop (padding = {})".format(crop_padding))
        tfm_train += [RandomCrop(cfg.INPUT.SIZE, padding=crop_padding)]

    if "random_resized_crop" in choices:
        print(f"+ random resized crop (size={cfg.INPUT.SIZE})")
        tfm_train += [
            RandomResizedCrop(cfg.INPUT.SIZE, interpolation=interp_mode)
        ]

    if "random_flip" in choices:
        print("+ random flip")
        tfm_train += [RandomHorizontalFlip()]

    if "imagenet_policy" in choices:
        print("+ imagenet policy")
        tfm_train += [ImageNetPolicy()]

    if "cifar10_policy" in choices:
        print("+ cifar10 policy")
        tfm_train += [CIFAR10Policy()]

    if "svhn_policy" in choices:
        print("+ svhn policy")
        tfm_train += [SVHNPolicy()]

    if "randaugment" in choices:
        n_ = cfg.INPUT.RANDAUGMENT_N
        m_ = cfg.INPUT.RANDAUGMENT_M
        print("+ randaugment (n={}, m={})".format(n_, m_))
        tfm_train += [RandAugment(n_, m_)]

    if "randaugment_fixmatch" in choices:
        n_ = cfg.INPUT.RANDAUGMENT_N
        print("+ randaugment_fixmatch (n={})".format(n_))
        tfm_train += [RandAugmentFixMatch(n_)]

    if "randaugment2" in choices:
        n_ = cfg.INPUT.RANDAUGMENT_N
        print("+ randaugment2 (n={})".format(n_))
        tfm_train += [RandAugment2(n_)]

    if "colorjitter" in choices:
        print("+ color jitter")
        tfm_train += [
            ColorJitter(
                brightness=cfg.INPUT.COLORJITTER_B,
                contrast=cfg.INPUT.COLORJITTER_C,
                saturation=cfg.INPUT.COLORJITTER_S,
                hue=cfg.INPUT.COLORJITTER_H,
            )
        ]

    if "randomgrayscale" in choices:
        print("+ random gray scale")
        tfm_train += [RandomGrayscale(p=cfg.INPUT.RGS_P)]

    if "gaussian_blur" in choices:
        print(f"+ gaussian blur (kernel={cfg.INPUT.GB_K})")
        tfm_train += [
            RandomApply([GaussianBlur(cfg.INPUT.GB_K)], p=cfg.INPUT.GB_P)
        ]

    print("+ to torch tensor of range [0, 1]")
    tfm_train += [ToTensor()]

    if "cutout" in choices:
        cutout_n = cfg.INPUT.CUTOUT_N
        cutout_len = cfg.INPUT.CUTOUT_LEN
        print("+ cutout (n_holes={}, length={})".format(cutout_n, cutout_len))
        tfm_train += [Cutout(cutout_n, cutout_len)]

    if "normalize" in choices:
        print(
            "+ normalization (mean={}, "
            "std={})".format(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
        )
        tfm_train += [normalize]

    if "gaussian_noise" in choices:
        print(
            "+ gaussian noise (mean={}, std={})".format(
                cfg.INPUT.GN_MEAN, cfg.INPUT.GN_STD
            )
        )
        tfm_train += [GaussianNoise(cfg.INPUT.GN_MEAN, cfg.INPUT.GN_STD)]

    if "instance_norm" in choices:
        print("+ instance normalization")
        tfm_train += [InstanceNormalization()]

    tfm_train = Compose(tfm_train)

    return tfm_train


def _build_transform_test(cfg, choices, target_size, normalize):
    print("Building transform_test")
    tfm_test = []

    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]

    print(f"+ resize the smaller edge to {max(cfg.INPUT.SIZE)}")
    tfm_test += [Resize(max(cfg.INPUT.SIZE), interpolation=interp_mode)]

    print(f"+ {target_size} center crop")
    tfm_test += [CenterCrop(cfg.INPUT.SIZE)]

    print("+ to torch tensor of range [0, 1]")
    tfm_test += [ToTensor()]

    if "normalize" in choices:
        print(
            "+ normalization (mean={}, "
            "std={})".format(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
        )
        tfm_test += [normalize]

    if "instance_norm" in choices:
        print("+ instance normalization")
        tfm_test += [InstanceNormalization()]

    tfm_test = Compose(tfm_test)

    return tfm_test

def to_tensor(cfg):
    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
    toTensor = []
    toTensor += [Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
    toTensor += [ToTensor()]
    if "normalize" in cfg.INPUT.TRANSFORMS:
        normalize = Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        toTensor += [normalize]
    return Compose(toTensor)