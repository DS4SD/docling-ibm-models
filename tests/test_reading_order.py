#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import os
import json

import numpy as np
import pytest
from PIL import Image

from huggingface_hub import snapshot_download

import docling_ibm_models.layoutmodel.layout_predictor as lp
from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor

from docling_ibm_models.reading_order.reading_order_rb import PageElement, ReadingOrderPredictor

@pytest.fixture(scope="module")
def init() -> dict:
    r"""
    Initialize the testing environment
    """
    # This config is missing the keys: "artifact_path", "info1.torch_file", "info2.torch_file"
    init = {
        "num_threads": 1,
        "test_imgs": [
            "tests/test_data/samples/ADS.2007.page_123.png",
        ],
        "info1": {
            "use_cpu_only": True,
            "image_size": 640,
            "threshold": 0.6,
        },
        "info2": {
            "use_cpu_only": True,
            "image_size": 640,
            "threshold": 0.6,
        },
        "pred_bboxes": 9,
    }

    # Download models from HF
    download_path = snapshot_download(repo_id="ds4sd/docling-models")
    artifact_path = os.path.join(download_path, "model_artifacts/layout/beehive_v0.0.5_pt")

    # Add the missing config keys
    init["artifact_path"] = artifact_path
    init["info1"]["torch_file"] = os.path.join(artifact_path, lp.MODEL_CHECKPOINT_FN)
    init["info2"]["torch_file"] = os.path.join(artifact_path, lp.MODEL_CHECKPOINT_FN)

    return init


def run_layoutpredictor(init: dict):
    r"""
    Unit test for the LayoutPredictor
    """
    # Initialize LayoutPredictor with envvars
    os.environ["USE_CPU_ONLY"] = ""
    os.environ["OMP_NUM_THREADS"] = "2"
    lpredictor = LayoutPredictor(init["artifact_path"])
    assert init["info1"] == lpredictor.info()

    # Initialize LayoutPredictor with optional parameters
    lpredictor = LayoutPredictor(
        init["artifact_path"], use_cpu_only=True
    )
    assert init["info2"] == lpredictor.info()

    # Unsupported input image
    is_exception = False
    try:
        for pred in lpredictor.predict("wrong"):
            pass
    except TypeError:
        is_exception = True
    assert is_exception

    # Predict on the test image
    for img_fn in init["test_imgs"]:
        with Image.open(img_fn) as img:
            # Load images as PIL objects
            for i, pred in enumerate(lpredictor.predict(img)):                
                print("PIL pred: {}".format(pred))
                yield pred

def test_readingorder():

    romodel = ReadingOrderPredictor()
    
    for pred in run_layoutpredictor(init): 
        print(pred.keys())

    assert True