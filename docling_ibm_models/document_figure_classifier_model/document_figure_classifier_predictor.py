#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import logging
from typing import List, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoConfig, AutoModelForImageClassification

_log = logging.getLogger(__name__)


class DocumentFigureClassifierPredictor:
    r"""
    Model for classifying document figures.

    Classifies figures as 1 out of 16 possible classes.

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
    _model : EfficientNetForImageClassification
        Pretrained EfficientNetb0 model.
    _image_processor : EfficientNetImageProcessor
        Processor for normalizing and preparing input images.
    _classes: List[str]:
        The classes used by the model.

    Methods
    -------
    __init__(artifacts_path, device, num_threads)
        Initializes the DocumentFigureClassifierPredictor with the specified parameters.
    info() -> dict:
        Retrieves configuration details of the DocumentFigureClassifierPredictor instance.
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

        self._image_processor = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.47853944, 0.4732864, 0.47434163],
                ),
            ]
        )

        config = AutoConfig.from_pretrained(artifacts_path)

        self._classes = list(config.id2label.values())
        self._classes.sort()

        _log.debug("CodeFormulaModel settings: {}".format(self.info()))

    def info(self) -> dict:
        """
        Retrieves configuration details of the DocumentFigureClassifierPredictor instance.

        Returns
        -------
        dict
            A dictionary containing configuration details such as the device,
            the number of threads used and the classe sused by the model.
        """
        info = {
            "device": self._device,
            "num_threads": self._num_threads,
            "classes": self._classes,
        }
        return info

    def predict(
        self, images: List[Union[Image.Image, np.ndarray]]
    ) -> List[Tuple[str, float, List[float]]]:
        r"""
            Performs inference on a batch of figures.

        Parameters
        ----------
        images : List[Union[Image.Image, np.ndarray]]
            A list of input images for inference. Each image can either be a
            PIL.Image.Image object or a NumPy array representing an image.

        Returns
        -------
        List[Tuple[str, float, List[float]]]
            A list of predictions, where each prediction is a tuple containing:
                - str: The predicted class name for the image.
                - float: The confidence score of the predicted class.
                - List[float]: The confidence scores for all possible classes.
        """
        processed_images = []
        for image in images:
            if isinstance(image, Image.Image):
                processed_images.append(image.convert("RGB"))
            elif isinstance(image, np.ndarray):
                processed_images.append(Image.fromarray(image).convert("RGB"))
            else:
                raise TypeError(
                    "Supported input formats are PIL.Image.Image or numpy.ndarray."
                )
        images = processed_images

        # (batch_size, 3, 224, 224)
        images = [self._image_processor(image) for image in images]
        images = torch.stack(images)

        with torch.no_grad():
            logits = self._model(images).logits  # (batch_size, num_classes)
            probs = logits.softmax(dim=1)  # (batch_size, num_classes)
            confs, preds = torch.max(probs, dim=1)  # (batch_size, )

            probs = probs.cpu().numpy().tolist()  # (batch_size, num_classes)
            preds = preds.cpu().numpy().tolist()  # (batch_size,)
            confs = confs.cpu().numpy().tolist()  # (batch_size,)

        predictions = [
            (self._classes[pred], float(conf), prob_dist)
            for pred, conf, prob_dist in zip(preds, confs, probs)
        ]

        return predictions
