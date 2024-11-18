#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import logging
import os
from collections.abc import Iterable
from typing import Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

_log = logging.getLogger(__name__)


MODEL_CHECKPOINT_FN = "model.pt"
DEFAULT_NUM_THREADS = 4


class LayoutPredictor:
    """
    Document layout prediction using torch
    """

    def __init__(
        self, artifact_path: str, device: str = "auto", num_threads: int = None
    ):
        """
        Provide the artifact path that contains the LayoutModel file

        Parameters
        ----------
        artifact_path: Path for the model torch file.
        device: (Optional) Device to run the inference. One of: ["cpu", "cuda", "mps", "auto"].
                           When it is "auto", the best available device is selected.
                           Default value is "auto"
        num_threads: (Optional) Number of threads to run the inference when the device is "cpu".


        Raises
        ------
        FileNotFoundError when the model's torch file is missing
        """
        # Initialize classes map:
        self._classes_map = {
            0: "background",
            1: "Caption",
            2: "Footnote",
            3: "Formula",
            4: "List-item",
            5: "Page-footer",
            6: "Page-header",
            7: "Picture",
            8: "Section-header",
            9: "Table",
            10: "Text",
            11: "Title",
            12: "Document Index",
            13: "Code",
            14: "Checkbox-Selected",
            15: "Checkbox-Unselected",
            16: "Form",
            17: "Key-Value Region",
        }

        # Blacklisted classes
        self._black_classes = set(["Form", "Key-Value Region"])

        # Set basic params
        self._threshold = 0.6  # Score threshold
        self._image_size = 640
        self._size = np.asarray([[self._image_size, self._image_size]], dtype=np.int64)

        # Set device based on init parameter or availability
        device_name = device.lower()
        if device_name in ["cuda", "mps", "cpu"]:
            self._device = torch.device(device_name)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        # Model file
        self._torch_fn = os.path.join(artifact_path, MODEL_CHECKPOINT_FN)
        if not os.path.isfile(self._torch_fn):
            raise FileNotFoundError("Missing torch file: {}".format(self._torch_fn))

        # Set number of threads for CPU
        if self._device.type == "cpu":
            self._num_threads = num_threads
            torch.set_num_threads(self._num_threads)

        # Load model and move to device
        self._model = torch.jit.load(self._torch_fn, map_location=self._device)
        self._model.eval()

        _log.debug("LayoutPredictor settings: {}".format(self.info()))

    def info(self) -> dict:
        """
        Get information about the configuration of LayoutPredictor
        """
        info = {
            "torch_file": self._torch_fn,
            "device": str(self._device),
            "image_size": self._image_size,
            "threshold": self._threshold,
        }
        return info

    @torch.inference_mode()
    def predict(self, orig_img: Union[Image.Image, np.ndarray]) -> Iterable[dict]:
        """
        Predict bounding boxes for a given image.
        The origin (0, 0) is the top-left corner and the predicted bbox coords are provided as:
        [left, top, right, bottom]

        Parameter
        ---------
        origin_img: Image to be predicted as a PIL Image object or numpy array.

        Yield
        -----
        Bounding box as a dict with the keys: "label", "confidence", "l", "t", "r", "b"

        Raises
        ------
        TypeError when the input image is not supported
        """
        # Convert image format
        if isinstance(orig_img, Image.Image):
            page_img = orig_img.convert("RGB")
        elif isinstance(orig_img, np.ndarray):
            page_img = Image.fromarray(orig_img).convert("RGB")
        else:
            raise TypeError("Not supported input image format")

        w, h = page_img.size
        orig_size = torch.tensor([w, h], device=self._device)[None]

        transforms = T.Compose(
            [
                T.Resize((640, 640)),
                T.ToTensor(),
            ]
        )
        img = transforms(page_img)[None].to(self._device)

        # Predict
        labels, boxes, scores = self._model(img, orig_size)

        # Yield output
        for label_idx, box, score in zip(labels[0], boxes[0], scores[0]):
            # Filter out blacklisted classes
            label_idx = int(label_idx.item())
            score = float(score.item())
            label = self._classes_map[label_idx + 1]
            if label in self._black_classes:
                continue

            # Check against threshold
            if score > self._threshold:
                bbox_float = [float(b.item()) for b in box]
                l = min(w, max(0, bbox_float[0]))
                t = min(h, max(0, bbox_float[1]))
                r = min(w, max(0, bbox_float[2]))
                b = min(h, max(0, bbox_float[3]))
                yield {
                    "l": l,
                    "t": t,
                    "r": r,
                    "b": b,
                    "label": label,
                    "confidence": score,
                }
