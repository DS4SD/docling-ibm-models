#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
from __future__ import division

import collections
import numbers
import random

import torch

from docling_ibm_models.tableformer.data_management import functional as F


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Attention: The multiprocessing used in dataloader of pytorch
    is not friendly with lambda function in Windows
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        # assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd
        # if 'Windows' in platform.system():
        #     raise RuntimeError("Can't pickle lambda funciton in windows system")

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class RandomTransforms(object):
    """Base class for a list of transformations with randomness
    Args:
        transforms (list or tuple): list of transformations
    """

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomChoice(RandomTransforms):
    """Apply single transformation randomly picked from a list"""

    def __call__(self, img, target):
        t = random.choice(self.transforms)
        return t(img, target)


class RandomCrop(object):
    def __init__(self, size, margin_crop):
        self.size = list(size)
        self.margin_crop = list(margin_crop)
        # margin_crop: w, h

    def __call__(self, img, target):
        # img (w,h,ch)
        image_height, image_width = img.shape[0], img.shape[1]
        """
        img (np.ndarray): Image to be cropped.
        x: Upper pixel coordinate.
        y: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        """
        if image_width > 0 and image_height > 0:
            cropped_image = F.crop(
                img,
                self.margin_crop[1],
                self.margin_crop[0],
                image_height - (self.margin_crop[1] * 2),
                image_width - (self.margin_crop[0] * 2),
            )

            target_ = target.copy()
            target_["boxes"][:, 0] = target_["boxes"][:, 0] - self.margin_crop[0]
            target_["boxes"][:, 1] = target_["boxes"][:, 1] - self.margin_crop[1]
            target_["boxes"][:, 2] = target_["boxes"][:, 2] - self.margin_crop[0]
            target_["boxes"][:, 3] = target_["boxes"][:, 3] - self.margin_crop[1]
        else:
            cropped_image = img
        return cropped_image, target_


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        pad_x1 = random.randint(0, self.max_pad)
        pad_y1 = random.randint(0, self.max_pad)
        img = img.copy()
        padded_image = F.pad(img, (pad_x, pad_y, pad_x1, pad_y1), fill=(255, 255, 255))
        target_ = target.copy()
        if target["boxes"] is not None:
            target_["boxes"][:, 0] = target_["boxes"][:, 0] + pad_x
            target_["boxes"][:, 1] = target_["boxes"][:, 1] + pad_y
            target_["boxes"][:, 2] = target_["boxes"][:, 2] + pad_x
            target_["boxes"][:, 3] = target_["boxes"][:, 3] + pad_y
        return padded_image, target_


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):

        assert isinstance(brightness, float) or (
            isinstance(brightness, collections.Iterable) and len(brightness) == 2
        )
        assert isinstance(contrast, float) or (
            isinstance(contrast, collections.Iterable) and len(contrast) == 2
        )
        assert isinstance(saturation, float) or (
            isinstance(saturation, collections.Iterable) and len(saturation) == 2
        )
        assert isinstance(hue, float) or (
            isinstance(hue, collections.Iterable) and len(hue) == 2
        )

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if isinstance(brightness, numbers.Number):

            if brightness > 0:
                brightness_factor = random.uniform(
                    max(0, 1 - brightness), 1 + brightness
                )
                transforms.append(
                    Lambda(lambda img: F.adjust_brightness(img, brightness_factor))
                )

            if contrast > 0:
                contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
                transforms.append(
                    Lambda(lambda img: F.adjust_contrast(img, contrast_factor))
                )

            if saturation > 0:
                saturation_factor = random.uniform(
                    max(0, 1 - saturation), 1 + saturation
                )
                transforms.append(
                    Lambda(lambda img: F.adjust_saturation(img, saturation_factor))
                )

            if hue > 0:
                hue_factor = random.uniform(-hue, hue)
                transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        else:

            if brightness[0] > 0 and brightness[1] > 0:

                brightness_factor = random.uniform(brightness[0], brightness[1])
                transforms.append(
                    Lambda(lambda img: F.adjust_brightness(img, brightness_factor))
                )

            if contrast[0] > 0 and contrast[1] > 0:

                contrast_factor = random.uniform(contrast[0], contrast[1])
                transforms.append(
                    Lambda(lambda img: F.adjust_contrast(img, contrast_factor))
                )

            if saturation[0] > 0 and saturation[1] > 0:

                saturation_factor = random.uniform(saturation[0], saturation[1])
                transforms.append(
                    Lambda(lambda img: F.adjust_saturation(img, saturation_factor))
                )

            if hue[0] > 0 and hue[1] > 0:
                hue_factor = random.uniform(hue[0], hue[1])
                transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = ComposeSingle(transforms)

        return transform

    def __call__(self, img, target):
        """
        Args:
            img (np.ndarray): Input image.
        Returns:
            np.ndarray: Color jittered image.
        """
        transform = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )
        return transform(img), target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "brightness={0}".format(self.brightness)
        format_string += ", contrast={0}".format(self.contrast)
        format_string += ", saturation={0}".format(self.saturation)
        format_string += ", hue={0})".format(self.hue)
        return format_string


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, target=None):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor, self.mean, self.std), target

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class NoTransformation(object):
    """Do Nothing"""

    def __call__(self, img, target):
        return img, target


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ComposeSingle(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
        (h, w), output size will be matched to this. If size is an int,
        smaller edge of the image will be matched to this number.
        i.e, if height > width, then image will be rescaled to
        (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
        ``BILINEAR``
    """

    def __init__(self, size, interpolation="BILINEAR"):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, target=None):
        """
        Args:
        img (np.ndarray): Image to be scaled.
        Returns:
        np.ndarray: Rescaled image.
        """
        # Resize bboxes (in pixels)
        x_scale = 0
        y_scale = 0

        if img.shape[1] > 0:
            x_scale = self.size[0] / img.shape[1]
        if img.shape[0] > 0:
            y_scale = self.size[1] / img.shape[0]

        # loop over bboxes
        if target is not None:
            if target["boxes"] is not None:
                target_ = target.copy()
                target_["boxes"][:, 0] = x_scale * target_["boxes"][:, 0]
                target_["boxes"][:, 1] = y_scale * target_["boxes"][:, 1]
                target_["boxes"][:, 2] = x_scale * target_["boxes"][:, 2]
                target_["boxes"][:, 3] = y_scale * target_["boxes"][:, 3]
        return F.resize(img, self.size, self.interpolation), target

    def __repr__(self):
        interpolate_str = self.interpolation
        return self.__class__.__name__ + "(size={0}, interpolation={1})".format(
            self.size, interpolate_str
        )
