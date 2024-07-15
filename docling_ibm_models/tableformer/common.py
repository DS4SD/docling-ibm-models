#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import argparse
import json
import logging
import os

import torch

import docling_ibm_models.tableformer.settings as s
from docling_ibm_models.tableformer.models.common.base_model import BaseModel

LOG_LEVEL = logging.DEBUG
logger = s.get_custom_logger("common", LOG_LEVEL)


def validate_config(config):
    r"""
    Validate the provided configuration file.
    A ValueError exception will be thrown in case the config file is invalid

    Parameters
    ----------
    config : dictionary
        Configuration for the tablemodel

    Returns
    -------
    bool : True on success
    """
    if "model" not in config:
        return True
    if "preparation" not in config:
        return True
    assert (
        "max_tag_len" in config["preparation"]
    ), "Config error: 'preparation.max_tag_len' parameter is missing"
    if "seq_len" in config["model"]:
        assert (
            config["model"]["seq_len"] > 0
        ), "Config error: 'model.seq_len' should be positive"
        assert config["model"]["seq_len"] <= (
            config["preparation"]["max_tag_len"] + 2
        ), "Config error: 'model.seq_len' should be up to 'preparation.max_tag_len' + 2"

    return True


def parse_arguments():
    r"""
    Parse the input arguments
    A ValueError exception will be thrown in case the config file is invalid
    """
    parser = argparse.ArgumentParser(description="Train the TableModel")
    parser.add_argument(
        "-c", "--config", required=True, default=None, help="configuration file (JSON)"
    )
    args = parser.parse_args()
    config_filename = args.config

    assert os.path.isfile(config_filename), "FAILURE: Config file not found."
    return read_config(config_filename)


def read_config(config_filename):
    with open(config_filename, "r") as fd:
        config = json.load(fd)

    # Validate the config file
    validate_config(config)

    return config


def safe_get_parameter(input_dict, index_path, default=None, required=False):
    r"""
    Safe get parameter from a nested dictionary.

    Provide a nested dictionary (dictionary of dictionaries) and a list of indices:
    - If the whole index path exists the value pointed by it is returned
    - Otherwise the default value is returned.

    Input:
        input_dict: Data structure of nested dictionaries.
        index_path: List with the indices path to follow inside the input_dict.
        default: Default value to return if the indices path is broken.
        required: If true a ValueError exception will be raised in case the parameter does not exist
    Output:
        The value pointed by the index path or "default".
    """
    if input_dict is None or index_path is None:
        return default

    d = input_dict
    for i in index_path[:-1]:
        if i not in d:
            if required:
                raise ValueError("Missing parameter: {}".format(i))
            return default
        d = d[i]

    last_index = index_path[-1]
    if last_index not in d:
        if required:
            raise ValueError("Missing parameter: {}".format(last_index))
        return default

    return d[last_index]


def get_prepared_data_filename(prepared_data_part, dataset_name):
    r"""
    Build the full filename of the prepared data part

    Parameters
    ----------
    prepared_data_part : string
        Part of the prepared data
    dataset_name : string
        Name of the dataset

    Returns
    -------
    string
        The full filename for the prepared file
    """
    template = s.PREPARED_DATA_PARTS[prepared_data_part]
    if "<POSTFIX>" in template:
        template = template.replace("<POSTFIX>", dataset_name)
    return template


def create_dataset_and_model(config, purpose, fixed_padding=False):
    r"""
    Gets a model from configuration

    Parameters
    ---------
    config : Dictionary
        The configuration of the model
    purpose : string
        One of "train", "eval", "predict"
    fixed_padding : bool
        Parameter passed to the constructor of the DataLoader

    Returns
    -------
    In case a Model cannot be initialized return None, None, None. Otherwise:

    device : selected device
    dataset : Instance of the DataLoader
    model : Instance of the model
    """
    from docling_ibm_models.tableformer.data_management.tf_dataset import TFDataset

    model_type = config["model"]["type"]
    model = None

    # Get env vars:
    use_cpu_only = os.environ.get("USE_CPU_ONLY", False)
    use_cuda_only = not use_cpu_only

    # Use the cpu for the evaluation
    device = "cpu"  # Default, run on CPU
    num_gpus = torch.cuda.device_count()  # Check if GPU is available
    if use_cuda_only:
        device = "cuda:0" if num_gpus > 0 else "cpu"  # Run on first available GPU
    else:
        device = "cpu"

    # Create the DataLoader
    # loader = DataLoader(config, purpose, fixed_padding=fixed_padding)
    dataset = TFDataset(config, purpose, fixed_padding=fixed_padding)
    dataset.set_device(device)
    dataset_val = None
    if config["train"]["validation"] and purpose == "train":
        dataset_val = TFDataset(config, "val", fixed_padding=fixed_padding)
        dataset_val.set_device(device)
    if model_type == "TableModel04_rs":
        from docling_ibm_models.tableformer.models.table04_rs.tablemodel04_rs import (  # noqa: F401
            TableModel04_rs,
        )
    # Find the model class and create an instance of it
    for candidate in BaseModel.__subclasses__():
        if candidate.__name__ == model_type:
            init_data = dataset.get_init_data()
            model = candidate(config, init_data, purpose, device)

    if model is None:
        logger.warn("Not found model: " + str(model_type))
        return None, None, None

    logger.info("Found model: " + str(model_type))

    if purpose == s.PREDICT_PURPOSE:
        return device, dataset, model
    else:
        return device, dataset, dataset_val, model
