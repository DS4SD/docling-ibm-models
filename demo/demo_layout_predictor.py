#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor

ARTIFACT_PATH = "tests/test_data/model_artifacts"


def demo(
    logger: logging.Logger,
    artifact_path: str,
    num_threads: int,
    img_dir: str,
    viz_dir: str,
):
    r"""
    Apply LayoutPredictor on the input image directory

    If you want to load from PDF:
    pdf_image = pyvips.Image.new_from_file("test_data/ADS.2007.page_123.pdf", page=0)
    """
    # Create the layout predictor
    lpredictor = LayoutPredictor(artifact_path, num_threads=num_threads)
    logger.info("LayoutPredictor settings: {}".format(lpredictor.info()))

    # Predict all test png images
    for img_fn in Path(img_dir).rglob("*.png"):
        logger.info("Predicting '%s'...", img_fn)
        start_t = time.time()

        with Image.open(img_fn) as image:
            # Predict layout
            preds = list(lpredictor.predict(image))
            dt_ms = 1000 * (time.time() - start_t)
            logger.debug("Time elapsed for prediction(ms): %s", dt_ms)

            # Draw predictions
            out_img = image.copy()
            draw = ImageDraw.Draw(out_img)

            for i, pred in enumerate(preds):
                scr = pred["confidence"]
                lab = pred["label"]
                box = [
                    round(pred["l"]),
                    round(pred["t"]),
                    round(pred["r"]),
                    round(pred["b"]),
                ]

                if lab == "Table":
                    draw.rectangle(
                        box,
                        outline="red",
                    )
                    draw.text(
                        (box[0], box[1]),
                        text=str(lab),
                        fill="blue",
                    )
                    logger.info("Table %s: bbox=%s", i, box)

            save_fn = os.path.join(viz_dir, os.path.basename(img_fn))
            out_img.save(save_fn)
            logger.info("Saving prediction visualization in: '%s'", save_fn)


def main(args):
    r""" """
    num_threads = int(args.num_threads) if args.num_threads is not None else None
    img_dir = args.img_dir
    viz_dir = args.viz_dir

    # Initialize logger
    logger = logging.getLogger("LayoutPredictor")
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Ensure the viz dir
    Path(viz_dir).mkdir(parents=True, exist_ok=True)

    # Test the LayoutPredictor
    demo(logger, ARTIFACT_PATH, num_threads, img_dir, viz_dir)


if __name__ == "__main__":
    r"""
    python -m demo.demo_layout_predictor -i <images_dir>
    """
    parser = argparse.ArgumentParser(description="Test the LayoutPredictor")
    parser.add_argument(
        "-n", "--num_threads", required=False, default=None, help="Number of threads"
    )
    parser.add_argument(
        "-i",
        "--img_dir",
        required=True,
        help="PNG images input directory",
    )
    parser.add_argument(
        "-v",
        "--viz_dir",
        required=False,
        default="viz/",
        help="Directory to save prediction visualizations",
    )

    args = parser.parse_args()
    main(args)
