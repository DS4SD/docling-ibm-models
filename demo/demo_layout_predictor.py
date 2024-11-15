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
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import snapshot_download

# TODO: Switch LayoutModel implementations
# from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor
from docling_ibm_models.layoutmodel.layout_predictor_jit import LayoutPredictor


def save_predictions(prefix: str, viz_dir: str, img_fn: str, img, predictions: dict):
    img_path = Path(img_fn)

    image = img.copy()
    draw = ImageDraw.Draw(image)

    predictions_filename = f"{prefix}_{img_path.stem}.txt"
    predictions_fn = os.path.join(viz_dir, predictions_filename)
    with open(predictions_fn, "w") as fd:
        for pred in predictions:
            bbox = [
                round(pred["l"].item(), 2),
                round(pred["t"].item(), 2),
                round(pred["r"].item(), 2),
                round(pred["b"].item(), 2),
            ]
            label = pred["label"]
            confidence = round(pred["confidence"], 3)

            # Save the predictions in txt file
            pred_txt = f"{prefix} {img_fn}: {label} - {bbox} - {confidence}\n"
            fd.write(pred_txt)

            # Draw the bbox and label
            draw.rectangle(bbox, outline="orange")
            txt = f"{pred["label"]}: {round(pred["confidence"], 2)}"
            draw.text((bbox[0], bbox[1]), text=txt, font=ImageFont.load_default(), fill="blue")

    draw_filename = f"{prefix}_{img_path.name}"
    draw_fn = os.path.join(viz_dir, draw_filename)
    image.save(draw_fn)


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

            # Save predictions
            logger.info("Saving prediction visualization in: '%s'", viz_dir)
            save_predictions("ST", viz_dir, img_fn, image, preds)


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

    # TODO: Switch LayoutModel implementations
    # Download models from HF
    # download_path = snapshot_download(repo_id="ds4sd/docling-models")
    # artifact_path = os.path.join(download_path, "model_artifacts/layout/beehive_v0.0.5_pt")

    os.environ["TORCH_DEVICE"] = "cpu"
    artifact_path = "/Users/nli/data/models/layout_model/online_docling_models/v2.0.1"

    # artifact_path = "/Users/nli/data/models/layout_model/safe_tensors"

    # Test the LayoutPredictor
    demo(logger, artifact_path, num_threads, img_dir, viz_dir)


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
