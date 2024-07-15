#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import glob
import os

import numpy as np
from PIL import Image

import docling_ibm_models.tableformer.common as c
from docling_ibm_models.tableformer.data_management.data_transformer import (
    DataTransformer,
)


def dump_np(img_np: np.array, fn, n=6):
    # Expect to receive a numpy array for an image with the shape [channels, rows, columns]
    s = img_np.shape
    if s[0] not in [1, 2, 3, 4] or len(s) != 3:
        print("Image of invalid shape: {}".format(s))
        return

    channels = s[0]
    rows = s[1]
    cols = s[2]
    w = n + 6
    with open(fn, "w") as fd:
        for r in range(rows):
            for col in range(cols):
                for ch in range(channels):
                    x = img_np[ch][r][col]
                    if isinstance(x, np.float32):
                        f_str = "0:>{}.{}f".format(w, n)
                    elif isinstance(x, np.uint8):
                        f_str = "0:>{}".format(w)
                    else:
                        return False

                    x_str = ("{" + f_str + "}").format(x)
                    fd.write(x_str)
                    if ch < channels - 1:
                        fd.write(" ")
                fd.write("\n")
    return True


def dump_channels(save_dir, fn_prefix, img_np: np.array):
    # Dump the np array into 3 files per channel
    img_np_ch0 = img_np[0, :, :]
    img_np_ch1 = img_np[1, :, :]
    img_np_ch2 = img_np[2, :, :]
    txt_ch0_fn = os.path.join(save_dir, fn_prefix + "_ch0.txt")
    txt_ch1_fn = os.path.join(save_dir, fn_prefix + "_ch1.txt")
    txt_ch2_fn = os.path.join(save_dir, fn_prefix + "_ch2.txt")
    np.savetxt(txt_ch0_fn, img_np_ch0)
    np.savetxt(txt_ch1_fn, img_np_ch1)
    np.savetxt(txt_ch2_fn, img_np_ch2)
    print(f"{txt_ch0_fn}")
    print(f"{txt_ch1_fn}")
    print(f"{txt_ch2_fn}")


def prepare_image(config):
    transformer = DataTransformer(config)
    predict_dir = config["predict"]["predict_dir"]
    use_normalization = config["dataset"]["image_normalization"]["state"]

    pattern = os.path.join(predict_dir, "*.png")
    for img_fn in glob.glob(pattern):
        print(f"img_fn: {img_fn}")

        with Image.open(img_fn) as img:
            # Dump the initial image in txt files
            img_np = np.array(img)

            # Reshape the image in order to print it
            img_np_m = np.moveaxis(img_np, 2, 0)
            print(
                "orig. img_np.shape: {}, reshaped image: {}".format(
                    img_np.shape, img_np_m.shape
                )
            )
            original_fn = img_fn + "_python.txt"
            dump_np(img_np_m, original_fn)

            r_img_ten = transformer.rescale_in_memory(img, use_normalization)
            print("npimgc: {} - {}".format(r_img_ten.type(), r_img_ten.size()))

            # Dump the processed image tensor in txt files
            r_img_np = r_img_ten.numpy()

            prepared_fn = img_fn + "_python_prepared.txt"
            dump_np(r_img_np, prepared_fn)


if __name__ == "__main__":
    config = c.parse_arguments()
    prepare_image(config)
