#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import logging

import docling_ibm_models.tableformer.common as c
from docling_ibm_models.tableformer.data_management.tf_dataset import TFDataset

LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG


def dataset_test(config):
    r"""
    Parameters
    ----------
    config : dictionary
        The configuration settings
    """

    # model_type = config["model"]["type"]
    # Create the device and the Dataset
    device = "cpu"
    dataset = TFDataset(config, "train")
    dataset.set_device(device)

    # Loop over the data
    dataset.reset()
    dataset.shuffle()
    for i, batch in enumerate(dataset):
        print("Loading batch: {}".format(i))


if __name__ == "__main__":
    config = c.parse_arguments()
    dataset_test(config)
