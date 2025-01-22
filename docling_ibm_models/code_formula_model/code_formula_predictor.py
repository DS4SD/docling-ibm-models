#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import logging
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

from docling_ibm_models.code_formula_model.models.sam_opt import SamOPTForCausalLM
from docling_ibm_models.code_formula_model.models.sam_opt_image_processor import (
    SamOptImageProcessor,
)

_log = logging.getLogger(__name__)


class CodeFormulaPredictor:
    """
    Code and Formula Predictor using a multi-modal vision-language model.

    This class enables the prediction of code or LaTeX representations
    from input images of code snippets or mathematical formulas.

    Attributes
    ----------
    _device : str
        The device on which the model is loaded (e.g., 'cpu' or 'cuda').
    _num_threads : int
        Number of threads used for inference when running on CPU.
    _tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for processing textual inputs to the model.
    _model : transformers.PreTrainedModel
        Pretrained multi-modal vision-language model.
    _image_processor : transformers.ImageProcessor
        Processor for normalizing and preparing input images.
    _temperature : float
        Sampling temperature for generation; controls randomness in predictions.
    """

    def __init__(
        self,
        artifacts_path: str,
        device: str = "cpu",
        num_threads: int = 4,
    ):
        """
        Initializes the CodeFormulaPredictor with the specified model artifacts.

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

        self._tokenizer = AutoTokenizer.from_pretrained(
            artifacts_path, use_fast=True, padding_side="left"
        )
        self._model = SamOPTForCausalLM.from_pretrained(artifacts_path).to(self._device)
        self._model.eval()

        self._image_processor = SamOptImageProcessor.from_pretrained(artifacts_path)

        _log.debug("CodeFormulaModel settings: {}".format(self.info()))

    def info(self) -> dict:
        """
        Retrieves configuration details of the CodeFormulaPredictor instance.

        Returns
        -------
        dict
            A dictionary containing configuration details such as the device and
            the number of threads used.
        """
        info = {
            "device": self._device,
            "num_threads": self._num_threads,
        }
        return info

    def _get_prompt(self, label: str) -> str:
        """
        Constructs the prompt for the model based on the input label.

        Parameters
        ----------
        label : str
            The type of input, either 'code' or 'formula'.

        Returns
        -------
        str
            The constructed prompt including necessary tokens and query.

        Raises
        ------
        NotImplementedError
            If the label is not 'code' or 'formula'.
        """
        if label == "code":
            query = "<code_image_to_text>"
        elif label == "formula":
            query = "<equation>"
        else:
            raise NotImplementedError("Label must be either code or formula")

        prompt = (
            "A chat between a curious user and an artificial intelligence"
            " assistant. The assistant gives helpful, detailed, and polite answers to"
            " the user's questions. USER: "
        )
        prompt += (
            "<img>" + "<imgpad>" * 256 + "</img>" + "\n" + " ASSISTANT:" + "\n" + query
        )

        return prompt

    @torch.inference_mode()
    def predict(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        labels: List[str],
        temperature: float = 0.1,
    ) -> List[str]:
        """
        Predicts the textual representation of input images (code or LaTeX).

        Parameters
        ----------
        images : List[Union[Image.Image, np.ndarray]]
            List of images to be processed, provided as PIL Image objects or numpy arrays.
        labels : List[str]
            List of labels indicating the type of each image ('code' or 'formula').
        temperature : float, optional
            Sampling temperature for generation, by default set to 0.1.

        Returns
        -------
        List[str]
            List of predicted textual outputs for each input image in the given input
            order.

        Raises
        ------
        TypeError
            If any of the input images is not of a supported type (PIL Image or numpy array).
        Excpetion
            In case the temperature is an invalid number.
        """
        if (type(temperature) != float and type(temperature) != int) or temperature < 0:
            raise Exception("Temperature must be a number greater or equal to 0.")

        do_sample = True
        if temperature == 0:
            do_sample = False
            temperature = None

        if len(labels) != len(images):
            raise Exception(
                "The number of images must be the same as the number of labels."
            )

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

        images_tensor = torch.stack([self._image_processor(img) for img in images]).to(
            self._device
        )

        prompts = [self._get_prompt(label) for label in labels]

        tokenized = self._tokenizer(prompts, padding=True, return_tensors="pt")
        tokenized = {k: v.to(self._device) for k, v in tokenized.items()}

        prompt_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        if self._device == "cpu":
            output_ids_list = self._model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                images=images_tensor,
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=4096 - prompt_ids.shape[1],
                use_cache=True,
            )
        else:
            with torch.autocast(device_type=self._device, dtype=torch.bfloat16):
                output_ids_list = self._model.generate(
                    prompt_ids,
                    images=images_tensor,
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=4096 - prompt_ids.shape[1],
                    use_cache=True,
                )

        outputs = self._tokenizer.batch_decode(
            output_ids_list[:, prompt_ids.shape[1] :], skip_special_tokens=True
        )

        return outputs
