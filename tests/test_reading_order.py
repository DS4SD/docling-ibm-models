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

from datasets import load_dataset

from typing import List
import random

from docling_ibm_models.reading_order.reading_order_rb import PageElement, ReadingOrderPredictor

from docling_core.types.doc.document import DoclingDocument, DocItem

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def rank_array(arr):
    """Compute ranks, resolving ties by averaging."""
    sorted_indices = sorted(range(len(arr)), key=lambda i: arr[i])  # Sort indices
    ranks = [0] * len(arr)  # Initialize ranks
    
    i = 0
    while i < len(arr):
        start = i
        while i + 1 < len(arr) and arr[sorted_indices[i]] == arr[sorted_indices[i + 1]]:
            i += 1  # Handle ties
        avg_rank = sum(range(start + 1, i + 2)) / (i - start + 1)  # Average rank for ties
        for j in range(start, i + 1):
            ranks[sorted_indices[j]] = avg_rank
        i += 1
    return ranks

def spearman_rank_correlation(arr1, arr2):
    assert len(arr1) == len(arr2), "Arrays must have the same length"
    
    # Compute ranks
    rank1 = rank_array(arr1)
    rank2 = rank_array(arr2)

    # Compute rank differences and apply formula
    d = [rank1[i] - rank2[i] for i in range(len(arr1))]
    d_squared_sum = sum(d_i ** 2 for d_i in d)

    n = len(arr1)
    rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
    
    return rho
            
def test_readingorder():

    # Init the reading-order model
    romodel = ReadingOrderPredictor()

    ds = load_dataset("ds4sd/docling-dpbench")
    for row in ds["test"]:
        true_doc = DoclingDocument.model_validate_json(row["GroundTruthDocument"])
        
        true_elements: List[PageElement] = []
        pred_elements: List[PageElement] = []
        
        for item, level in true_doc.iterate_items():
            if isinstance(item, DocItem):
                for prov in item.prov:

                    page_height = true_doc.pages[prov.page_no].size.height
                    bbox = prov.bbox.to_bottom_left_origin(page_height=page_height)
                    
                    true_elements.append(
                        PageElement(
                            cid=len(true_elements),
                            page_no=prov.page_no,
                            page_size = true_doc.pages[prov.page_no].size,
                            label=item.label,
                            l = bbox.l,
                            r = bbox.r,
                            b = bbox.b,
                            t = bbox.t,
                            coord_origin = bbox.coord_origin
                        )
                    )

        rand_elements = copy.deepcopy(true_elements)
        random.shuffle(rand_elements)

        """
        print(f"reading {os.path.basename(filename)}")        
        for true_elem, rand_elem in zip(true_elements, rand_elements):
            print("true: ", str(true_elem), ", rand: ", str(rand_elem))
        """
        
        pred_elements = romodel.predict_reading_order(page_elements=rand_elements)
        #pred_elements = romodel.predict_page(page_elements=rand_elements)    

        assert len(pred_elements)==len(true_elements), f"{len(pred_elements)}!={len(true_elements)}"

        true_cids, pred_cids = [], []
        for true_elem, pred_elem, rand_elem in zip(true_elements,
                                                   pred_elements,
                                                   rand_elements):
            true_cids.append(true_elem.cid)
            pred_cids.append(pred_elem.cid)

        score = spearman_rank_correlation(true_cids, pred_cids)

        filename = row["document_id"]

        # Identify special cases ...
        if filename in ["doc_906d54a21ef3c7bfac03f4bb613b0c79ef32fdf81b362450c79e98a96f88708a_page_000001.png",
                        "doc_2cd17a32ee330a239e19c915738df0c27e8ec3635a60a7e16e2a0cf3868d4af3_page_000001.png",
                        "doc_bcb3dafc35b5e7476fd1b9cd6eccf5eeef936cd5b13ad846a4943f1e7797f4e9_page_000001.png",
                        "doc_a0edae1fa147c7bb78ebc493743a68ba4372b5ead31f2a2b146c35119462379e_page_000001.png",
                        "doc_94ba5468fcb6277721947697048846dc0d0551296be3b45f5918ab857d21dcc7_page_000001.png",
                        "doc_cbb4a13ffd01d9f777fdb939451d6a21cea1b869ee50d79581451e3601df9ec8_page_000001.png"]:
            # print(f"{os.path.basename(filename)}: {score}")
            assert score>=0.60, f"reading-order score={score}>0.60"            
        else:
            assert score>=0.90, f"reading-order score={score}>0.90"
            
