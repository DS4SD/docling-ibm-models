#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import numbers
from collections.abc import Iterable, Sequence

import cv2
import numpy as np
import torch
from torchvision.transforms import functional

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

INTER_MODE = {
    "NEAREST": cv2.INTER_NEAREST,
    "BILINEAR": cv2.INTER_LINEAR,
    "BICUBIC": cv2.INTER_CUBIC,
}

PAD_MOD = {
    "constant": cv2.BORDER_CONSTANT,
    "edge": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_DEFAULT,
    "symmetric": cv2.BORDER_REFLECT,
}


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_tensor(pic):
    """Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    Args:
        pic (np.ndarray, torch.Tensor): Image to be converted to tensor, (H x W x C[RGB]).
    Returns:
        Tensor: Converted image.
    """

    if _is_numpy_image(pic):
        if len(pic.shape) == 2:
            pic = cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor) or img.max() > 1:
            return img.float().div(255)
        else:
            return img
    elif _is_tensor_image(pic):
        return pic

    else:
        try:
            return to_tensor(np.array(pic))
        except Exception:
            raise TypeError("pic should be ndarray. Got {}".format(type(pic)))


def to_cv_image(pic, mode=None):
    """Convert a tensor to an ndarray.
    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (str): color space and pixel depth of input data (optional)
                    for example: cv2.COLOR_RGB2BGR.
    Returns:
        np.array: Image converted to PIL Image.
    """
    if not (_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError("pic should be Tensor or ndarray. Got {}.".format(type(pic)))

    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.squeeze(np.transpose(pic.numpy(), (1, 2, 0)))

    if not isinstance(npimg, np.ndarray):
        raise TypeError("Input pic must be a torch.Tensor or NumPy ndarray")
    if mode is None:
        return npimg

    else:
        return cv2.cvtColor(npimg, mode)


def normalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.
    See ``Normalize`` for more details.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.
    Returns:
        Tensor: Normalized Tensor image.
    """
    if _is_tensor_image(tensor):
        for t, m, s in zip(tensor, mean, std, strict=False):
            t.sub_(m).div_(s)
        return tensor
    elif _is_numpy_image(tensor):
        return (tensor.astype(np.float32) - 255.0 * np.array(mean)) / np.array(std)
    else:
        raise RuntimeError("Undefined type")


def resize(img, size, interpolation="BILINEAR"):
    """Resize the input CV Image to the given size.
    Args:
        img (np.ndarray): Image to be resized.
        size (tuple or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (str, optional): Desired interpolation. Default is ``BILINEAR``
    Returns:
        cv Image: Resized image.
    """
    if not _is_numpy_image(img):
        raise TypeError("img should be CV Image. Got {}".format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError("Got inappropriate size arg: {}".format(size))

    if isinstance(size, int):
        h, w, c = img.shape
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(
                img, dsize=(ow, oh), interpolation=INTER_MODE[interpolation]
            )
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(
                img, dsize=(ow, oh), interpolation=INTER_MODE[interpolation]
            )
    else:
        oh, ow = size
        return cv2.resize(
            img, dsize=(int(ow), int(oh)), interpolation=INTER_MODE[interpolation]
        )


def to_rgb_bgr(pic):
    """Converts a color image stored in BGR sequence to RGB (BGR to RGB)
    or stored in RGB sequence to BGR (RGB to BGR).
    Args:
        pic (np.ndarray, torch.Tensor): Image to be converted, (H x W x 3).
    Returns:
        Tensor: Converted image.
    """

    if _is_numpy_image(pic) or _is_tensor_image(pic):
        img = pic[:, :, [2, 1, 0]]
        return img
    else:
        try:
            return to_rgb_bgr(np.array(pic))
        except Exception:
            raise TypeError("pic should be numpy.ndarray or torch.Tensor.")


def pad(img, padding, fill=(0, 0, 0), padding_mode="constant"):
    """Pad the given CV Image on all sides with speficified padding mode and fill value.
    Args:
        img (np.ndarray): Image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (int, tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric.
        Default is constant.
            constant: pads with a constant value, this value is specified with fill
            edge: pads with the last value on the edge of the image
            reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
            symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    Returns:
        CV Image: Padded image.
    """
    if not _is_numpy_image(img):
        raise TypeError("img should be CV Image. Got {}".format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError("Got inappropriate padding arg")
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError("Got inappropriate fill arg")
    if not isinstance(padding_mode, str):
        raise TypeError("Got inappropriate padding_mode arg")

    if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple")

    assert padding_mode in [
        "constant",
        "edge",
        "reflect",
        "symmetric",
    ], "Padding mode should be either constant, edge, reflect or symmetric"

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, Sequence) and len(padding) == 4:
        pad_left, pad_top, pad_right, pad_bottom = padding

    if isinstance(fill, numbers.Number):
        fill = (fill,) * (2 * len(img.shape) - 3)

    if padding_mode == "constant":
        assert (len(fill) == 3 and len(img.shape) == 3) or (
            len(fill) == 1 and len(img.shape) == 2
        ), "channel of image is {} but length of fill is {}".format(
            img.shape[-1], len(fill)
        )

    img = cv2.copyMakeBorder(
        src=img,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=PAD_MOD[padding_mode],
        value=fill,
    )
    return img


def crop(img, x, y, h, w):
    """Crop the given CV Image.
    Args:
        img (np.ndarray): Image to be cropped.
        x: Upper pixel coordinate.
        y: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        CV Image: Cropped image.
    """
    assert _is_numpy_image(img), "img should be CV Image. Got {}".format(type(img))
    assert h > 0 and w > 0, "h={} and w={} should greater than 0".format(h, w)

    x1, y1, x2, y2 = round(x), round(y), round(x + h), round(y + w)

    #     try:
    #         check_point1 = img[x1, y1, ...]
    #         check_point2 = img[x2-1, y2-1, ...]
    #     except IndexError:
    #         img = cv2.copyMakeBorder(img, - min(0, x1), max(x2 - img.shape[0], 0),
    #                                  -min(0, y1), max(y2 - img.shape[1], 0),
    #                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])
    #         y2 += -min(0, y1)
    #         y1 += -min(0, y1)
    #         x2 += -min(0, x1)
    #         x1 += -min(0, x1)
    #
    #     finally:
    #         return img[x1:x2, y1:y2, ...].copy()
    return img[x1:x2, y1:y2, ...].copy()


def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    h, w, _ = img.shape
    th, tw = output_size
    i = int(round((h - th) * 0.5))
    j = int(round((w - tw) * 0.5))
    return crop(img, i, j, th, tw)


def resized_crop(img, i, j, h, w, size, interpolation="BILINEAR"):
    """Crop the given CV Image and resize it to desired size. Notably used in RandomResizedCrop.
    Args:
        img (np.ndarray): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics as ``scale``.
        interpolation (str, optional): Desired interpolation. Default is
            ``BILINEAR``.
    Returns:
        np.ndarray: Cropped image.
    """
    assert _is_numpy_image(img), "img should be CV Image"
    img = crop(img, i, j, h, w)
    img = resize(img, size, interpolation)
    return img


def hflip(img):
    """Horizontally flip the given PIL Image.
    Args:
        img (np.ndarray): Image to be flipped.
    Returns:
        np.ndarray:  Horizontall flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError("img should be CV Image. Got {}".format(type(img)))

    return cv2.flip(img, 1)


def vflip(img):
    """Vertically flip the given PIL Image.
    Args:
        img (CV Image): Image to be flipped.
    Returns:
        PIL Image:  Vertically flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    return cv2.flip(img, 0)


def five_crop(img, size):
    """Crop the given CV Image into four corners and the central crop.
    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.
    Returns:
        tuple: tuple (tl, tr, bl, br, center) corresponding top left,
            top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    h, w, _ = img.shape
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError(
            "Requested crop size {} is bigger than input size {}".format(size, (h, w))
        )
    tl = crop(img, 0, 0, crop_h, crop_w)
    tr = crop(img, 0, w - crop_w, crop_h, crop_w)
    bl = crop(img, h - crop_h, 0, crop_h, crop_w)
    br = crop(img, h - crop_h, w - crop_w, crop_h, crop_w)
    center = center_crop(img, (crop_h, crop_w))
    return (tl, tr, bl, br, center)


def ten_crop(img, size, vertical_flip=False):
    """Crop the given CV Image into four corners and the central crop plus the
       flipped version of these (horizontal flipping is used by default).
    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
       Args:
           size (sequence or int): Desired output size of the crop. If size is an
               int instead of sequence like (h, w), a square crop (size, size) is
               made.
           vertical_flip (bool): Use vertical flipping instead of horizontal
        Returns:
            tuple: tuple (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip,
                br_flip, center_flip) corresponding top left, top right,
                bottom left, bottom right and center crop and same for the
                flipped image.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    first_five = five_crop(img, size)

    if vertical_flip:
        img = vflip(img)
    else:
        img = hflip(img)

    second_five = five_crop(img, size)
    return first_five + second_five


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
        img (np.ndarray): CV Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        np.ndarray: Brightness adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError("img should be CV Image. Got {}".format(type(img)))

    im = img.astype(np.float32) * brightness_factor
    im = im.clip(min=0, max=255)
    return im.astype(img.dtype)


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.
    Args:
        img (np.ndarray): CV Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        np.ndarray: Contrast adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError("img should be CV Image. Got {}".format(type(img)))
    im = img.astype(np.float32)
    mean = round(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).mean())
    im = (1 - contrast_factor) * mean + contrast_factor * im
    im = im.clip(min=0, max=255)
    return im.astype(img.dtype)


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.
    Args:
        img (np.ndarray): CV Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a gray image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
    Returns:
        np.ndarray: Saturation adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    im = img.astype(np.float32)
    degenerate = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    im = (1 - saturation_factor) * degenerate + saturation_factor * im
    im = im.clip(min=0, max=255)
    return im.astype(img.dtype)


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See https://en.wikipedia.org/wiki/Hue for more details on Hue.
    Args:
        img (np.ndarray): CV Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
        np.ndarray: Hue adjusted image.
    """
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError("hue_factor is not in [-0.5, 0.5].")

    if not _is_numpy_image(img):
        raise TypeError("img should be CV Image. Got {}".format(type(img)))

    im = img.astype(np.uint8)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV_FULL)
    hsv[..., 0] += np.uint8(hue_factor * 255)

    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
    return im.astype(img.dtype)


def adjust_gamma(img, gamma, gain=1):
    """Perform gamma correction on an image.
    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:
        I_out = 255 * gain * ((I_in / 255) ** gamma)
    See https://en.wikipedia.org/wiki/Gamma_correction for more details.
    Args:
        img (np.ndarray): CV Image to be adjusted.
        gamma (float): Non negative real number. gamma larger than 1 make the
            shadows darker, while gamma smaller than 1 make dark regions
            lighter.
        gain (float): The constant multiplier.
    """
    if not _is_numpy_image(img):
        raise TypeError("img should be CV Image. Got {}".format(type(img)))

    if gamma < 0:
        raise ValueError("Gamma should be a non-negative real number")

    im = img.astype(np.float32)
    im = 255.0 * gain * np.power(im / 255.0, gamma)
    im = im.clip(min=0.0, max=255.0)
    return im.astype(img.dtype)


def to_grayscale(img, num_output_channels=1):
    """Convert image to grayscale version of image.
    Args:
        img (np.ndarray): Image to be converted to grayscale.
    Returns:
        CV Image:  Grayscale version of the image.
                    if num_output_channels == 1 : returned image is single channel
                    if num_output_channels == 3 : returned image is 3 channel with r == g == b
    """
    if not _is_numpy_image(img):
        raise TypeError("img should be CV Image. Got {}".format(type(img)))

    if num_output_channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif num_output_channels == 3:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError("num_output_channels should be either 1 or 3")

    return img


def gaussian_noise(img: np.ndarray, mean, std):
    imgtype = img.dtype
    gauss = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = np.clip((1 + gauss) * img.astype(np.float32), 0, 255)
    return noisy.astype(imgtype)


def poisson_noise(img):
    imgtype = img.dtype
    img = img.astype(np.float32) / 255.0
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = 255 * np.clip(
        np.random.poisson(img.astype(np.float32) * vals) / float(vals), 0, 1
    )
    return noisy.astype(imgtype)


def salt_and_pepper(img, prob=0.01):
    """Adds "Salt & Pepper" noise to an image.
    prob: probability (threshold) that controls level of noise
    """
    imgtype = img.dtype
    rnd = np.random.rand(img.shape[0], img.shape[1])
    noisy = img.copy()
    noisy[rnd < prob / 2] = 0.0
    noisy[rnd > 1 - prob / 2] = 255.0
    return noisy.astype(imgtype)


def cv_transform(img):
    img = salt_and_pepper(img)
    return to_tensor(img)


def pil_transform(img):
    return functional.to_tensor(img)
