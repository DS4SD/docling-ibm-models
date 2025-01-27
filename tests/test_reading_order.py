#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import os
import json
import glob
import copy

import logging

import numpy as np
import pytest
from PIL import Image

from typing import List
import random

from huggingface_hub import snapshot_download

import docling_ibm_models.layoutmodel.layout_predictor as lp
from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor

from docling_ibm_models.reading_order.reading_order_rb import PageElement, ReadingOrderPredictor

from docling_core.types.doc.document import DoclingDocument, DocItem

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@pytest.fixture(scope="module")
def init() -> dict:
    r"""
    Initialize the testing environment
    """
    # This config is missing the keys: "artifact_path", "info1.torch_file", "info2.torch_file"
    init = {
        "num_threads": 1,
        "test_imgs": sorted(glob.glob("tests/test_data/samples/*.png")),
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

def _test_readingorder(init: dict):
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

    # Init the reading-order model
    romodel = ReadingOrderPredictor()
    
    # Predict on the test image
    for img_fn in init["test_imgs"]:
        print(img_fn)
        
        with Image.open(img_fn) as img:
            pred_layout=[]

            # Load images as PIL objects
            for i, pred in enumerate(lpredictor.predict(img)):                
                pred_layout.append({
                    "label": pred["label"],
                    "t": pred["t"].item(),
                    "b": pred["b"].item(),
                    "l": pred["l"].item(),
                    "r": pred["r"].item(),
                })
            print(json.dumps(pred_layout, indent=2))

            page_elements = []
            for cid, item in enumerate(pred_layout):
                page_elements.append(PageElement(cid=cid, pid=0,
                                                 x0=item["l"], y0=item["r"],
                                                 x1=item["b"], y1=item["t"],
                                                 label=item["label"]))

            print(page_elements)
                
            ordered_elements = romodel.predict_page(page_elements)

            print(ordered_elements)


def test_readingorder():

    # Init the reading-order model
    romodel = ReadingOrderPredictor()
    
    filenames = sorted(glob.glob("/Users/taa/Documents/projects/docling-eval/benchmarks/DPBench-annotations-v03/json_annotations/*.json"))

    print(f"#-filenames: {len(filenames)}")
    
    for filename in filenames:
        print(f"reading {os.path.basename(filename)}")

        true_doc = DoclingDocument.load_from_json(filename=filename)

        true_elements: Dict[int, List[PageElement]] = {}
        pred_elements: Dict[int, List[PageElement]] = {}
        
        for item, level in true_doc.iterate_items():
            if isinstance(item, DocItem):
                for prov in item.prov:

                    print(f"\t{level}, {item.label}")
                    
                    if prov.page_no not in true_elements:
                        true_elements[prov.page_no] = []
                        pred_elements[prov.page_no] = []
                        
                    page_height = true_doc.pages[prov.page_no].size.height
                    
                    true_elements[prov.page_no].append(
                        PageElement(
                            cid=len(true_elements[prov.page_no]),
                            pid=0, 
                            label=item.label,
                            bbox=prov.bbox.to_bottom_left_origin(page_height=page_height)
                        )
                    )

        rand_elements = copy.deepcopy(true_elements)
        for page_no,val in rand_elements.items():
            random.shuffle(rand_elements[page_no])
        
            pred_elements[page_no], to_captions, to_footnotes = romodel.predict_page(page_elements=rand_elements[page_no])    

            print(f"#-true elems: {len(true_elements[page_no])}, #-pred elems: {len(pred_elements[page_no])}")

            print("to_captions: \n", to_captions)
            print("to_footnotes: \n", to_footnotes)
            
            assert len(pred_elements[page_no])==len(true_elements[page_no]), f"len(pred_elements[page_no])==len(true_elements[page_no]), {len(pred_elements[page_no])}=={len(true_elements[page_no])}"

            for true_elem, pred_elem, rand_elem in zip(true_elements[page_no],
                                                       pred_elements[page_no],
                                                       rand_elements[page_no]):
                print("true: ", str(true_elem), ", pred: ", str(pred_elem), ", rand: ", str(rand_elem))
                # assert true_elem.cid==pred_elem.cid

            
            
