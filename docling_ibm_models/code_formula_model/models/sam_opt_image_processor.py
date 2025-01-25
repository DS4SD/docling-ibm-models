#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
from PIL import Image
from torchvision.transforms import functional as F
from transformers import AutoImageProcessor
from transformers.image_processing_utils import ImageProcessingMixin


class SamOptImageProcessor(ImageProcessingMixin):

    def __init__(self, size=(1024, 1024), mean=None, std=None, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.mean = mean
        self.std = std

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image")

        image = F.resize(image, self.size)
        image = F.to_tensor(image)

        image = F.normalize(image, mean=self.mean, std=self.std)

        return image


AutoImageProcessor.register(SamOptImageProcessor, SamOptImageProcessor)
