#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import os
from collections.abc import Iterable
from typing import Union

from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

DEFAULT_NUM_THREADS = 4


class LayoutPredictor:
    """
    Document layout prediction using safe tensors
    """

    def __init__(self, artifact_path: str, num_threads: int = None):
        """
        Provide the artifact path that contains the LayoutModel file

        The number of threads is decided, in the following order, by:
        1. The init method parameter `num_threads`, if it is set.
        2. The envvar "OMP_NUM_THREADS", if it is set.
        3. The default value DEFAULT_NUM_THREADS.

        The execution device is decided by the env var "TORCH_DEVICE" with values:
        'cpu', 'cuda', or 'mps'. If not set, automatically selects the best available device.

        Parameters
        ----------
        artifact_path: Path for the model torch file.
        num_threads: (Optional) Number of threads to run the inference.

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

        # Set device based on env var or availability
        device_name = os.environ.get("TORCH_DEVICE", "").lower()
        if device_name in ["cuda", "mps", "cpu"]:
            self._device = torch.device(device_name)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        # Set number of threads for CPU
        if self._device.type == "cpu":
            if num_threads is None:
                num_threads = int(
                    os.environ.get("OMP_NUM_THREADS", DEFAULT_NUM_THREADS)
                )
            self._num_threads = num_threads
            torch.set_num_threads(self._num_threads)

        # Model file and configurations
        self._st_fn = os.path.join(artifact_path, "model.safetensors")
        if not os.path.isfile(self._st_fn):
            raise FileNotFoundError("Missing safe tensors file: {}".format(self._st_fn))

        # Load model and move to device
        processor_config = os.path.join(artifact_path, "preprocessor_config.json")
        model_config = os.path.join(artifact_path, "config.json")
        self._image_processor = RTDetrImageProcessor.from_json_file(processor_config)
        self._model = RTDetrForObjectDetection.from_pretrained(
            artifact_path, config=model_config, device_map=self._device
        )
        self._model.eval()

    def info(self) -> dict:
        """
        Get information about the configuration of LayoutPredictor
        """
        info = {
            "safe_tensors_file": self._st_fn,
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

        resize = {"height": self._image_size, "width": self._image_size}
        inputs = self._image_processor(
            images=page_img,
            return_tensors="pt",
            size=resize,
        ).to(self._device)
        outputs = self._model(**inputs)
        results = self._image_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([page_img.size[::-1]]),
            threshold=self._threshold,
        )

        w, h = page_img.size

        result = results[0]
        for score, label_id, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):
            score = float(score.item())

            label_id = int(label_id.item()) + 1  # Advance the label_id
            label_str = self._classes_map[label_id]

            # Filter out blacklisted classes
            if label_str in self._black_classes:
                continue

            l = min(w, max(0, box[0]))
            t = min(h, max(0, box[1]))
            r = min(w, max(0, box[2]))
            b = min(h, max(0, box[3]))
            yield {
                "l": l,
                "t": t,
                "r": r,
                "b": b,
                "label": label_str,
                "confidence": score,
            }
