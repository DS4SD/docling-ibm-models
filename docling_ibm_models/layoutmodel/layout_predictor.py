#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import os
from collections.abc import Iterable
from typing import Union

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_CHECKPOINT_FN = "model.pt"
DEFAULT_NUM_THREADS = 4


class LayoutPredictor:
    r"""
    Document layout prediction using ONNX
    """

    def __init__(
        self, artifact_path: str, num_threads: int = None, use_cpu_only: bool = False
    ):
        r"""
        Provide the artifact path that contains the LayoutModel ONNX file

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
        artifact_path: Path for the model ONNX file.
        num_threads: (Optional) Number of threads to run the inference.
        use_cpu_only: (Optional) If True, it forces CPU as the execution provider.

        Raises
        ------
        FileNotFoundError when the model's ONNX file is missing
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

        # Get env vars
        self._use_cpu_only = use_cpu_only or ("USE_CPU_ONLY" in os.environ)
        if num_threads is None:
            num_threads = int(os.environ.get("OMP_NUM_THREADS", DEFAULT_NUM_THREADS))
        self._num_threads = num_threads

        # Decide the execution providers
        if (
            not self._use_cpu_only
            and "CUDAExecutionProvider" in ort.get_available_providers()
        ):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self._providers = providers

        # Model ONNX file
        self._onnx_fn = os.path.join(artifact_path, MODEL_CHECKPOINT_FN)
        if not os.path.isfile(self._onnx_fn):
            raise FileNotFoundError("Missing ONNX file: {}".format(self._onnx_fn))

        # ONNX options
        self._options = ort.SessionOptions()
        self._options.intra_op_num_threads = self._num_threads
        self.sess = ort.InferenceSession(
            self._onnx_fn,
            sess_options=self._options,
            providers=self._providers,
        )

    def info(self) -> dict:
        r"""
        Get information about the configuration of LayoutPredictor
        """
        info = {
            "onnx_file": self._onnx_fn,
            "intra_op_num_threads": self._num_threads,
            "use_cpu_only": self._use_cpu_only,
            "providers": self._providers,
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
        page_img = page_img.resize((self._image_size, self._image_size))
        page_data = np.array(page_img, dtype=np.uint8) / np.float32(255.0)
        page_data = np.expand_dims(np.transpose(page_data, axes=[2, 0, 1]), axis=0)

        # Predict
        labels, boxes, scores = self.sess.run(
            output_names=None,
            input_feed={
                "images": page_data,
                "orig_target_sizes": self._size,
            },
        )

        # Yield output
        for label_idx, box, score in zip(labels[0], boxes[0], scores[0]):
            # Filter out blacklisted classes
            label = self._classes_map[label_idx]
            if label in self._black_classes:
                continue

            # Check against threshold
            if score > self._threshold:
                yield {
                    "l": box[0] / self._image_size * w,
                    "t": box[1] / self._image_size * h,
                    "r": box[2] / self._image_size * w,
                    "b": box[3] / self._image_size * h,
                    "label": label,
                    "confidence": score,
                }
