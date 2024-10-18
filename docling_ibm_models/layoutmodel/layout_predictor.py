#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import os
from collections.abc import Iterable
from typing import Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

MODEL_CHECKPOINT_FN = "model.pt"
DEFAULT_NUM_THREADS = 4


class LayoutPredictor:
    r"""
    Document layout prediction using torch
    """

    def __init__(
        self, artifact_path: str, num_threads: int = None, use_cpu_only: bool = False
    ):
        r"""
        Provide the artifact path that contains the LayoutModel file

        The number of threads is decided, in the following order, by:
        1. The init method parameter `num_threads`, if it is set.
        2. The envvar "OMP_NUM_THREADS", if it is set.
        3. The default value DEFAULT_NUM_THREADS.

        The execution provided is decided, in the following order:
        1. If the init method parameter `cpu_only` is True or the envvar "USE_CPU_ONLY" is set,
           it uses the "CPUExecutionProvider".
        3. Otherwise if the "CUDAExecutionProvider" is present, use:
            ["CUDAExecutionProvider", "CPUExecutionProvider"]:

        Parameters
        ----------
        artifact_path: Path for the model torch file.
        num_threads: (Optional) Number of threads to run the inference.
        use_cpu_only: (Optional) If True, it forces CPU as the execution provider.

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
        self._use_cpu_only = use_cpu_only or ("USE_CPU_ONLY" in os.environ)

        # Model file
        self._torch_fn = os.path.join(artifact_path, MODEL_CHECKPOINT_FN)
        if not os.path.isfile(self._torch_fn):
            raise FileNotFoundError("Missing torch file: {}".format(self._torch_fn))

        # Get env vars
        if num_threads is None:
            num_threads = int(os.environ.get("OMP_NUM_THREADS", DEFAULT_NUM_THREADS))
        self._num_threads = num_threads

        self.model = torch.jit.load(self._torch_fn)

    def info(self) -> dict:
        r"""
        Get information about the configuration of LayoutPredictor
        """
        info = {
            "torch_file": self._torch_fn,
            "use_cpu_only": self._use_cpu_only,
            "image_size": self._image_size,
            "threshold": self._threshold,
        }
        return info

    def predict(self, orig_img: Union[Image.Image, np.ndarray]) -> Iterable[dict]:
        r"""
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
        orig_size = torch.tensor([w, h])[None]

        transforms = T.Compose(
            [
                T.Resize((640, 640)),
                T.ToTensor(),
            ]
        )
        img = transforms(page_img)[None]
        # Predict
        with torch.no_grad():
            labels, boxes, scores = self.model(img, orig_size)

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
                l = min(w, max(0, box[0]))
                t = min(h, max(0, box[1]))
                r = min(w, max(0, box[2]))
                b = min(h, max(0, box[3]))
                yield {
                    "l": l,
                    "t": t,
                    "r": r,
                    "b": b,
                    "label": label,
                    "confidence": score,
                }
