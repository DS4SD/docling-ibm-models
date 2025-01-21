#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import logging
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageClassification, AutoProcessor

_log = logging.getLogger(__name__)


class DocumentFigureClassifierPredictor:
    r"""
    Model for classifying document figures.

    Classifies every figure in every page as 1 out of 16 possible classes.

    The classes are:
        1. "bar_chart"
        2. "bar_code"
        3. "chemistry_markush_structure"
        4. "chemistry_molecular_structure"
        5. "flow_chart"
        6. "icon"
        7. "line_chart"
        8. "logo"
        9. "map"
        10. "other"
        11. "pie_chart"
        12. "qr_code"
        13. "remote_sensing"
        14. "screenshot"
        15. "signature"
        16. "stamp"

    Attributes
    ----------
    _device : str
        The device on which the model is loaded (e.g., 'cpu' or 'cuda').
    _num_threads : int
        Number of threads used for inference when running on CPU.
    _model : transformers.PreTrainedModel
        Pretrained EfficientNetb0 model.
    _image_processor : transformers.ImageProcessor
        Processor for normalizing and preparing input images.

    Methods
    -------
    __init__(artifacts_path, device, num_threads)
        Initializes the DocumentFigureClassifierPredictor with the specified parameters.
    predict(images) -> List[List[float]]
        The confidence scores for the classification of each image.
    """

    def __init__(
        self,
        artifacts_path: str,
        device: str = "cpu",
        num_threads: int = 4,
    ):
        r"""
        Initializes the DocumentFigureClassifierPredictor.

        Parameters
        ----------
        artifacts_path : str
            Path to the directory containing the pretrained model files.
        device : str, optional
            Device to run the inference on ('cpu' or 'cuda'), by default "cpu".
        num_threads : int, optional
            Number of threads for CPU inference, by default 4.
        """
        self._device = device
        self._num_threads = num_threads

        if device == "cpu":
            torch.set_num_threads(self._num_threads)

        model = AutoModelForImageClassification.from_pretrained(artifacts_path)
        self._model = model.to(device)
        self._model.eval()

        self._image_processor = AutoProcessor.from_pretrained(artifacts_path)

        _log.debug("CodeFormulaModel settings: {}".format(self.info()))

    def predict(
        self, images: List[Union[Image.Image, np.ndarray]]
    ) -> List[List[float]]:
        r"""
        Performs inference on a batch of figures..

        Parameters
        ----------
        figures : List[Union[Image.Image, np.ndarray]]
            Input figures for inference.

        Returns
        -------
        List[List[float]]:
            The confidence scores for the classification of each image.
        """
        images_tmp = []
        for image in images:
            if isinstance(image, Image.Image):
                image = image.convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert("RGB")
            else:
                raise TypeError("Not supported input image format")
            images_tmp.append(image)
        images = images_tmp

        # (batch_size, 3, 224, 224)
        images = self._image_processor(images, return_tensor="pt")

        with torch.no_grad():
            logits = self._model(**images).logits  # (batch_size, num_classes)
            probs = F.softmax(logits, dim=1).cpu().numpy()  # (batch_size, num_classes)

        return probs
