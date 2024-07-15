#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import copy
import logging
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

import docling_ibm_models.tableformer.data_management.transforms as T
import docling_ibm_models.tableformer.settings as s

LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG


class DataTransformer:
    r"""
    Data transformations for the images and bboxes

    Check the "help" fields inside the config file for an explanation of each parameter
    """

    def __init__(self, config):
        self._config = config

        print("DataTransformer Init!")

    def _log(self):
        # Setup a custom logger
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    def append_id(self, filename):
        name, ext = os.path.splitext(filename)
        return "{name}_{uid}{ext}".format(name=name, uid="resized", ext=ext)

    def load_image(self, img_fn):
        r"""
        Load an image from the disk

        Parameters
        ----------
            img_fn: The filename of the image

        Returns
        -------
            PIL image object
        """

        img = Image.open(img_fn)
        return img

    def load_image_cv2(self, img_fn):
        r"""Load an image from the disk

        Parameters
        ----------
            img_fn: The filename of the image

        Returns
        -------
            CV2  image object
        """
        img = cv2.imread(img_fn)
        return img

    def save_image(self, img, img_fn):
        img.save(self.append_id(img_fn))

    def renderbboxes(self, img, bboxes):
        draw_img = ImageDraw.Draw(img)
        for i in range(len(bboxes)):
            draw_img.rectangle(bboxes[i], fill=None, outline=(255, 0, 0))
        return img

    def get_dataset_settings(self):
        dataset = {}
        debug = {"save_debug_images": False}

        if "dataset" in self._config:
            dataset = self._config["dataset"]
        if "debug" in self._config:
            debug = self._config["debug"]

        return dataset, debug

    def _prepare_image_from_file(self, image_fn, bboxes, convert_box=True):
        r"""
        Load the image from file and prepare it

        Parameters
        ----------
        image_fn : string
            Filename to load the image
        bboxes : dict
            Bounding boxes of the image
        convert_box : bool
            If true the bboxes are converted to xcycwh format

        Returns
        -------
        PIL image
            A PIL image object with the image prepared according to the settings in the config file
        bboxes : dict
            Converted bboxes of the image
        """
        im = self.load_image(image_fn)
        return self._prepare_image(im, bboxes, convert_box, image_fn)

    def _prepare_image(self, im, bboxes, convert_box=True, image_fn=None):
        r"""
        Parameters
        ----------
        im : PIL image object
        bboxes : dict
            Bounding boxes of the image
        convert_box : bool
            If true the bboxes are converted to xcycwh format
        image_fn : string
            Filename of the original image or None. It is used to save augmented image for debugging

        Returns
        -------
        PIL image
            A PIL image object with the image prepared according to the settings in the config file
        bboxes : dict
            Converted bboxes of the image
        """
        debug_settings = False
        settings, debug_settings = self.get_dataset_settings()

        desired_size = settings["resized_image"]
        old_size = im.size

        # Calculate Aspect Ratio if needed
        if settings["keep_AR"]:
            ratio = float(desired_size) / max(old_size)
        else:
            ratio = 1  # Square image

        new_size = old_size
        if max(old_size) < desired_size:
            # Image is smaller than desired
            # Upscale?
            if settings["up_scaling_enabled"]:
                # Calculate new image size, taking into account aspect ratio
                new_size = tuple([int(x * ratio) for x in old_size])
            else:
                new_size = old_size
        else:
            if settings["keep_AR"]:
                new_size = tuple([int(x * ratio) for x in old_size])
            else:
                new_size = [desired_size, desired_size]

        if not settings["keep_AR"]:
            if settings["up_scaling_enabled"]:
                new_size = [desired_size, desired_size]

        ######################################################################################
        # Use OpenCV to resize the image
        #
        # im = im.resize(new_size, Image.ANTIALIAS)

        import cv2

        np_im = np.array(im)
        np_resized = cv2.resize(np_im, new_size, interpolation=cv2.INTER_LANCZOS4)
        im = Image.fromarray(np_resized)
        ######################################################################################

        new_bboxes = copy.deepcopy(bboxes)

        # Resize bboxes (in pixels)
        x_scale = new_size[0] / old_size[0]
        y_scale = new_size[1] / old_size[1]
        # loop over bboxes
        for i in range(len(new_bboxes)):
            new_bboxes[i][0] = x_scale * bboxes[i][0]
            new_bboxes[i][1] = y_scale * bboxes[i][1]
            new_bboxes[i][2] = x_scale * bboxes[i][2]
            new_bboxes[i][3] = y_scale * bboxes[i][3]

        # Set background color for padding
        br = settings["padding_color"][0]
        bg = settings["padding_color"][1]
        bb = settings["padding_color"][2]
        bcolor = (br, bg, bb)
        # Create empty canvas of background color and desired size
        new_im = Image.new(mode="RGB", size=(desired_size, desired_size), color=bcolor)

        if "grayscale" in settings:
            if settings["grayscale"]:
                im = im.convert("LA")

        if settings["padding_mode"] == "frame":
            # If paddinds are around image, paste resized image in the center
            x_pad = (desired_size - new_size[0]) // 2
            y_pad = (desired_size - new_size[1]) // 2
            # Paste rescaled image
            new_im.paste(im, (x_pad, y_pad))
            # Offset (pad) bboxes
            # loop over bboxes
            for i in range(len(new_bboxes)):
                new_bboxes[i][0] += x_pad
                new_bboxes[i][1] += y_pad
                new_bboxes[i][2] += x_pad
                new_bboxes[i][3] += y_pad
        else:
            # Otherwise paste in the 0,0 coordinates
            new_im.paste(im, (0, 0))

        if debug_settings:
            if debug_settings["save_debug_images"]:
                aug_im = self.renderbboxes(new_im, bboxes)
                if "grayscale" in settings:
                    if settings["grayscale"]:
                        aug_im = aug_im.convert("LA")
                self.save_image(aug_im, image_fn)
        if convert_box:
            bboxes = self.xyxy_to_xcycwh(new_bboxes, desired_size)
        return new_im, bboxes

    # convert bboxes from [x1, y1, x2, y2] format to [xc, yc, w, h] format
    def xyxy_to_xcycwh(self, bboxes, size):
        # Use the "dataset.bbox_format" parameter to decide which bbox format to use
        bbox_format = self._config["dataset"].get("bbox_format", "4plet")

        conv_bboxes = []
        for i in range(len(bboxes)):
            x1 = bboxes[i][0] / size  # X1
            y1 = bboxes[i][1] / size  # Y1
            x2 = bboxes[i][2] / size  # X2
            y2 = bboxes[i][3] / size  # Y2
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            bw = abs(x2 - x1)
            bh = abs(y2 - y1)

            if bbox_format == "5plet":
                cls = bboxes[i][4]
                conv_bboxes.append([xc, yc, bw, bh, cls])
            else:
                conv_bboxes.append([xc, yc, bw, bh])

        # conv_bboxes = bboxes
        return conv_bboxes

    def rescale_in_memory(self, image, normalization):
        r"""
        Receive image and escale it in memory

        Parameters
        ----------
        image : PIL image
            The image data to rescale
        normalization : dictionary
            The normalization information with the format:
                "state": "true or false if image normalization is to be enabled",
                "mean": "mean values to use if state is true",
                "std": "std values to use if state is true"
        Returns
        -------
        npimgc : FloatTensor
            The loaded and properly transformed image data
        """
        settings, debug_settings = self.get_dataset_settings()
        new_image, _ = self._prepare_image(image, {}, convert_box=False, image_fn=None)
        # Convert to nparray
        npimg = np.asarray(new_image)  # (width, height, channels)

        # Convert to float?
        npimgc = npimg.copy()

        # Transpose numpy array (image)
        npimgc = npimgc.transpose(2, 0, 1)  # (channels, width, height)
        npimgc = torch.FloatTensor(npimgc / 255.0)

        if normalization:
            transform = transforms.Compose(
                [
                    transforms.Normalize(
                        mean=self._config["dataset"]["image_normalization"]["mean"],
                        std=self._config["dataset"]["image_normalization"]["std"],
                    )
                ]
            )
            npimgc = transform(npimgc)

        return npimgc

    def _rescale(self, image_fn, bboxes, normalization):
        r"""
        Rescale, resize, pad the given image and its associated bboxes according to the settings
        from the config

        Parameters
        ----------
        image_fn: full image file name
        bboxes: List with bboxes in the format [x1, y1, x2, y2] with box's top-right,
                bottom-left points
        statistics: Dictionary with statistics over the whole image dataset.
                    The keys are: "mean", "variance", "std" and each value is a list with the
                    coresponding statistical value for each channel. Normally there are 3
                    channels.

        Returns
        -------
        npimgc : FloatTensor
            The loaded and properly transformed image data
        bboxes: List with bboxes in the format [xc, yc, w, h] where xc, yc are the coords of the
                 center, w, h the width and height of the bbox and all are normalized to the
                 scaled size of the image

        Raises
        ------
        ValueError
            In case the configuration and the image dimensions make it impossible to rescale the
            image throw a ValueError exception
        """
        settings, debug_settings = self.get_dataset_settings()
        # new_image is a PIL object
        new_image, new_bboxes = self._prepare_image_from_file(image_fn, bboxes)
        # Convert to nparray
        npimg = np.asarray(new_image)  # (width, height, channels)
        # Convert to float?
        npimgc = npimg.copy()
        # Transpose numpy array (image)
        npimgc = npimgc.transpose(2, 0, 1)  # (channels, width, height)
        npimgc = torch.FloatTensor(npimgc / 255.0)

        if normalization:
            transform = transforms.Compose(
                [
                    transforms.Normalize(
                        mean=self._config["dataset"]["image_normalization"]["mean"],
                        std=self._config["dataset"]["image_normalization"]["std"],
                    )
                ]
            )
            npimgc = transform(npimgc)
        return npimgc, new_bboxes

    def rescale_old(self, image, bboxes, statistics=None):
        r"""
        Rescale, resize, pad the given image and its associated bboxes according to the settings
        from the config

        Input:
            image: np array (channels, width, height)
            bboxes: List with bboxes in the format [x1, y1, x2, y2] with box's top-right,
                    bottom-left points
            statistics: Dictionary with statistics over the whole image dataset.
                        The keys are: "mean", "variance", "std" and each value is a list with the
                        coresponding statistical value for each channel. Normally there are 3
                        channels.

        Output:
            image: np array (channels, resized_image, resized_image)
            bboxes: List with bboxes in the format (xc, yc, w, h) where xc, yc are the coords of the
                     center, w, h the width and height of the bbox and all are normalized to the
                     scaled size of the image

        Exceptions:
            In case the configuration and the image dimensions make it impossible to rescale the
            image throw a ValueError exception
        """
        image_size = 448
        # Convert the image to (width, height, channels)
        image = image.transpose(1, 2, 0)

        # Convert to PIL format and resize
        image = Image.fromarray(image)
        image = image.resize((image_size, image_size), Image.ANTIALIAS)
        return image, bboxes

    def rescale_batch(self, images, bboxes, statistics=None):
        r"""
        Rescale, resize, pad the given batch of images and its associated bboxes according to the
        settings from the config.

        Input:
            images: np array (batch_size, channels, width, height)
            bboxes:
            statistics: Dictionary with statistics over the whole image dataset.
                        The keys are: "mean", "variance", "std" and each value is a list with the
                        coresponding statistical value for each channel. Normally there are 3
                        channels.

        Output:
            image batch: np array (batch_size, channels, resized_image, resized_image)
            bboxes:

        Exceptions:
            In case the configuration and the image dimensions make it impossible to rescale the
            image throw a ValueError exception
        """
        pass

    def sample_preprocessor(self, image_fn, bboxes, purpose, table_bboxes=None):
        r"""
        Rescale, resize, pad the given image and its associated bboxes according to the settings
        from the config

        Parameters
        ----------
        image_fn: full image file name
        bboxes: List with bboxes in the format [x1, y1, x2, y2] with box's top-right,
                bottom-left points
        statistics: Dictionary with statistics over the whole image dataset.
                    The keys are: "mean", "variance", "std" and each value is a list with the
                    coresponding statistical value for each channel. Normally there are 3
                    channels.

        Returns
        -------
        npimgc : FloatTensor
            The loaded and properly transformed image data
        bboxes: List with bboxes in the format [xc, yc, w, h] where xc, yc are the coords of the
                 center, w, h the width and height of the bbox and all are normalized to the
                 scaled size of the image

        Raises
        ------
        ValueError
            In case the configuration and the image dimensions make it impossible to rescale the
            image throw a ValueError exception
        """
        settings, debug_settings = self.get_dataset_settings()
        img = self.load_image_cv2(image_fn)
        img = np.ascontiguousarray(img)

        target = {
            "size": [img.shape[1], img.shape[2]],
            "boxes": (
                torch.from_numpy(np.array(bboxes)[:, :4])
                if purpose != s.PREDICT_PURPOSE
                else None
            ),
            "classes": (
                np.array(bboxes)[:, -1] if purpose != s.PREDICT_PURPOSE else None
            ),
            "area": img.shape[1] * img.shape[2],
        }

        optional_transforms = [T.NoTransformation()]

        # Necessary preprocessing ends here, experimental options begin below.
        # DETR format, might be necessary to keep this structure to share other functions used by
        # the community

        if purpose == s.TRAIN_PURPOSE:
            if self._config["dataset"]["color_jitter"]:
                jitter = T.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                )
                optional_transforms.append(jitter)

            if self._config["dataset"]["rand_pad"]:
                pad_val = random.randint(0, 50)
                rand_pad = T.RandomPad(pad_val)
                optional_transforms.append(rand_pad)

            if table_bboxes is not None:
                if self._config["dataset"]["rand_crop"]:
                    w_, h_, _ = img.shape[0], img.shape[1], img.shape[2]
                    w_c, h_c = table_bboxes[0], table_bboxes[1]
                    f_w, f_h = random.randint(0, w_c), random.randint(0, h_c)
                    rand_crop = T.RandomCrop((w_, h_), (f_w, f_h))
                    optional_transforms.append(rand_crop)

        # transform_opt = random.choice(optional_transforms)
        normalize = T.Normalize(
            mean=self._config["dataset"]["image_normalization"]["mean"],
            std=self._config["dataset"]["image_normalization"]["std"],
        )
        resized_size = self._config["dataset"]["resized_image"]
        resize = T.Resize([resized_size, resized_size])

        transformations = T.RandomChoice(optional_transforms)

        img, target = transformations(img, target)
        img, target = normalize(img, target)
        img, target = resize(img, target)

        img = img.transpose(2, 1, 0)  # (channels, width, height)
        img = torch.FloatTensor(img / 255.0)
        bboxes_ = target["boxes"]
        classes_ = target["classes"]
        desired_size = img.shape[1]

        if purpose != s.PREDICT_PURPOSE:
            bboxes_ = np.concatenate(
                (bboxes_, np.expand_dims(classes_, axis=1)), axis=1
            )
            bboxes_ = self.xyxy_to_xcycwh(bboxes_, desired_size)

        return img, bboxes_
