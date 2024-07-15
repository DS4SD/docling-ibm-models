#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import json
import logging
import os
from glob import glob
from html import escape

import jsonlines
import numpy as np
import torch
import torch.utils.data
from lxml import html
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

import docling_ibm_models.tableformer.common as c
import docling_ibm_models.tableformer.settings as s
import docling_ibm_models.tableformer.utils.utils as u
from docling_ibm_models.tableformer.data_management.data_transformer import (
    DataTransformer,
)

LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG


class TFDataset(Dataset):
    def __init__(self, config, purpose, fixed_padding=False):
        r"""
        Parameters
        ----------
        config : Dictionary
            The input configuration file
        purpose : string
            One of s.TRAIN_PURPOSE, s.VAL_PURPOSE, s.PREDICT_PURPOSE
        fixed_padding : bool
            If False (default), the produced tag sequences will be truncated to the maximum
            actual length among the tag sequences of the batch
            If True the produced tag and cell sequences will have fixed length equal to
            max_tag_len and max_cell_len respectively.
        """
        self.cml_task = {}
        self.cml_logger = {}
        self._config = config
        self._fixed_padding = fixed_padding
        self._index = 0  # Index to the current image file
        self._max_tag_len = c.safe_get_parameter(
            config, ["preparation", "max_tag_len"], required=True
        )
        self._max_cell_len = c.safe_get_parameter(
            config, ["preparation", "max_cell_len"], required=True
        )
        self._resized_image = c.safe_get_parameter(
            config, ["dataset", "resized_image"], required=True
        )
        self.annotation = c.safe_get_parameter(
            config, ["preparation", "annotation"], required=True
        )
        self._image_normalization = config["dataset"]["image_normalization"]

        self._load_cells = c.safe_get_parameter(
            config, ["dataset", "load_cells"], required=False
        )
        self._predict_dir = c.safe_get_parameter(
            config, ["predict", "predict_dir"], required=False
        )
        self._train_bbox = c.safe_get_parameter(
            config, ["train", "bbox"], required=False
        )
        self._predict_bbox = c.safe_get_parameter(
            config, ["predict", "bbox"], required=False
        )

        self._log().debug("purpose: {}".format(purpose))
        self._log().debug("resized_image: {}".format(self._resized_image))
        self._log().debug("image_normalization: {}".format(self._image_normalization))

        # Check the type of the dataset
        dataset_type = c.safe_get_parameter(config, ["dataset", "type"])
        if dataset_type not in s.supported_datasets:
            msg = "Unsupported dataset type: " + dataset_type
            self._log().error(msg)
            raise NotImplementedError(msg)
        self._dataset_type = dataset_type

        # Check the purpose of the object
        if purpose not in [
            s.TRAIN_PURPOSE,
            s.VAL_PURPOSE,
            s.TEST_PURPOSE,
            s.PREDICT_PURPOSE,
        ]:
            msg = "Unsupported purpose: " + purpose
            self._log().error(msg)
            raise Exception(msg)
        self._purpose = purpose

        # The batch_size is grounded to 1 in case of VAL, PREDICT
        if purpose == s.TRAIN_PURPOSE:
            self._batch_size = c.safe_get_parameter(
                config, ["train", "batch_size"], required=True
            )
        else:
            self._batch_size = 1
        self._log().debug("batch_size: {}".format(self._batch_size))

        self._transformer = DataTransformer(config)
        if purpose == s.PREDICT_PURPOSE:
            self._build_predict_cache()
        else:
            self._build_cache()

        self._index = 0  # Index to the current image file
        self._ind = np.array(range(self._dataset_size))

    def set_device(self, device):
        r"""
        Set the device to be used to place the tensors when looping over the data

        Parameters
        ----------
        device : torch.device or int
            The device to do the training
        """
        self._device = device

    def _log(self):
        # Setup a custom logger
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    def __len__(self):
        return int(self._dataset_size)

    def get_batch_num(self):
        return int(np.ceil(self._dataset_size / self._batch_size))

    def _pad_tensors_from_batch(self, batch, field):
        r"""
        Retrieves data about tensor - "field" from the raw DataLoader batch list
        And pads it to maximum batch length, to be further collated in the
        custom collator function

        Parameters
        ----------
        batch : list
            The list of samples, obtained by DataLoader from the sampler
            a non-collated batch
        field: string
            Name of the tensor data-point from the sample, that has to be collated,
            and because of that has to be padded to max length of the tensor in the batch

        Returns
        -------
        batchoftensors: list
            The list of padded tensors
        """
        # Make list of selected items by field name
        list_of_items = [x[field] for x in batch]
        tagseqlens = []
        # Identify lengths of tensor "lists"
        for i, ten in enumerate(list_of_items):
            tagnum = ten[0].size()[0]
            tagseqlens.append(tagnum)
        # Get the biggest length
        maxbatchtaglen = max(tagseqlens)

        # Prepare new list with padded tensors
        batchoftensors = []
        for i, ten in enumerate(list_of_items):
            tagtensor = ten
            newtagtensor = torch.zeros(
                1, maxbatchtaglen, dtype=torch.long, device=self._device
            )
            newtagtensor[:, : tagtensor.size()[1]] = tagtensor
            batchoftensors.append(newtagtensor)

        return batchoftensors

    def bcol(self, batch):
        r"""
        Custom collate fucntion for Pytorch DataLoader, to collate items prepared
        by TFDataset into batches

        Parameters
        ----------
        batch : list
            The list of samples, obtained by DataLoader from the sampler
            a non-collated batch

        Returns
        -------
        batchoftensors: tuple
            Tuple of lists of items, properly collated into batches
        """
        collated = {}
        test_gt = {}
        if bool(batch[0]["test_gt"]):
            for x in batch:
                test_gt.update(x["test_gt"])

        if len(batch) > 1:
            # In case batch length is more than 1 we want to collate all elements in batch
            # Every element has it's own rule how to collate

            if self._load_cells:
                cells = [
                    item for sublist in [x["cells"] for x in batch] for item in sublist
                ]
            else:
                cells = []
            cell_lens = [
                item for sublist in [x["cell_lens"] for x in batch] for item in sublist
            ]
            cell_bboxes = []

            if "cell_bboxes" in batch[0] and batch[0]["cell_bboxes"]:
                for x in batch:
                    cell_bboxes.append(
                        {
                            "boxes": x["cell_bboxes"][0][0],
                            "labels": x["cell_bboxes"][0][1],
                        }
                    )

            samples = [
                item for sublist in [x["samples"] for x in batch] for item in sublist
            ]

            # Sequences of tags have to be padded and then collated:
            batchoftags = self._pad_tensors_from_batch(batch, "tags")
            tags = pad_sequence(batchoftags, batch_first=True)
            tags = torch.squeeze(tags, 1)

            tag_lens = pad_sequence([x["tag_lens"] for x in batch], batch_first=True)
            tag_lens = torch.squeeze(tag_lens, 1)
            num_cells = pad_sequence([x["num_cells"] for x in batch], batch_first=True)
            num_cells = torch.squeeze(num_cells, 1)

            collated = (
                torch.cat([x["imgs"] for x in batch], dim=0),
                tags,
                tag_lens,
                num_cells,
                cells,
                cell_lens,
                cell_bboxes,
                samples,
                test_gt,
            )
        else:
            # In case batch length is 1 we just formulate the expected output
            cell_bboxes = None
            if "cell_bboxes" in batch[0] and batch[0]["cell_bboxes"]:
                cell_bboxes = {
                    "boxes": batch[0]["cell_bboxes"][0][0],
                    "labels": batch[0]["cell_bboxes"][0][1],
                }

            collated = (
                batch[0]["imgs"],
                batch[0]["tags"],
                batch[0]["tag_lens"],
                batch[0]["num_cells"],
                batch[0]["cells"],
                batch[0]["cell_lens"],
                cell_bboxes,
                batch[0]["samples"],
                test_gt,
            )
        return collated

    def __getitem__(self, idx):
        r"""
        Retrieve data by the specific index from the cache
        Required for Pytorch DataSampler, and to be used together with DataLoader
        Depending on the "purpose" different returned objects can be None.

        Returns
        (All data points presented in a dictionary, each wrapped into a list,
        for easier batching and collating later on)
        -------
        imgs : tensor (batch_size, image_channels, resized_image, resized_image)
            Batch with the rescaled images
        tags : The tags of the images. The object is one of:
            None : If purpose is not "train"
            (batch_size, max_tag_len + 2) : If purpose is "train" and "fixed_padding" is true.
                                            The +2 over the max_tag_len is for the <start> <stop>
            (batch_size, batch_max_tag_len) : If purpose is "train" and "fixed_padding" is false,
                                              where "batch_max_tag_len" is the max length of the
                                              tags in batch
        tag_lens : The real length of the tags per image in the batch. The object is one of:
            None : If purpose is not "train"
            (batch_size, 1) : If purpose is "train"
        num_cells : The number of cells per image in the batch. The object is one of:
            None : If purpose is not "train"
            (batch_size, 1) : If purpose is "train"
        cells : The cell tags for the images in the batch. The object is one of:
            None : If purpose is not "train"
            list with LongTensor: If purpose is "train"
        cell_lens : The length of the cell tags per image in the batch. The object is one of:
            None : If purpose is not "train"
            list with LongTensor: If purpose is "train"
        cell_bboxes : The transformed (rescaled, padded, etc) bboxes of the cells for all images in
                      batch. Each list is a bbox in the format [xc, yc, w, h] where xc, yc are the
                      coords of the center, w, h the width and height of the bbox and all are
                      normalized to the scaled size of the image. The object is one of:
            None : If purpose is not "train"
            list of lists: If purpose is "train"
        samples : list of string
            The filenames in the batch
        val_gt : The ground turth raw attributes for the validation split. The object is one of:
            None : If purpose is not "val"
            dictionary : If purpose is "val"
        """

        # Check if index out of bounds
        if idx >= self._dataset_size:
            return None

        # Move current _index to requested idx
        # self._index = idx

        # Images, __getitem__ provides only 1 image for specified index
        # (batch_size, image_channels, resized_image, resized_image)
        imgs = torch.zeros(
            1, self._img_ch, self._resized_image, self._resized_image, dtype=torch.float
        ).to(self._device)

        # Initialize all output objects to None
        # Depending on the "purpose" some of them will be populated
        tags = None
        tag_lens = None
        num_cells = None
        cells = None
        cell_lens = None
        cell_bboxes = None

        val_tags = None
        val_tag_lens = None
        val_num_cells = None
        val_cells = None
        val_cell_lens = None
        val_cell_bboxes = None

        test_gt = {}
        test_tags = None
        test_tag_lens = None

        # Train specific output
        if self._purpose == s.TRAIN_PURPOSE:
            tag_len = self._taglens[idx]
            fixed_tag_len = self._max_tag_len + 2  # <start>...<end>
            if self._fixed_padding:
                tags = torch.zeros(
                    1, fixed_tag_len, dtype=torch.long, device=self._device
                )
            else:
                tags = torch.zeros(1, tag_len, dtype=torch.long, device=self._device)

            tag_lens = torch.zeros(1, 1, dtype=torch.long).to(self._device)
            num_cells = torch.zeros(1, 1, dtype=torch.long).to(self._device)
            cells = []
            cell_lens = []
            cell_bboxes = []
        # val specific output
        elif self._purpose == s.VAL_PURPOSE:
            val_tag_len = self._val_taglens[idx]
            val_fixed_tag_len = self._max_tag_len + 2  # <start>...<end>
            if self._fixed_padding:
                val_tags = torch.zeros(
                    1, val_fixed_tag_len, dtype=torch.long, device=self._device
                )
            else:
                val_tags = torch.zeros(
                    1, val_tag_len, dtype=torch.long, device=self._device
                )

            val_tag_lens = torch.zeros(1, 1, dtype=torch.long).to(self._device)
            val_num_cells = torch.zeros(1, 1, dtype=torch.long).to(self._device)
            val_cells = []
            val_cell_lens = []
            val_cell_bboxes = []

        elif self._purpose == s.TEST_PURPOSE:
            if len(self._test_taglens) > 0:
                # Dictionary with the raw attributes for the groundtruth. Keys are the filenames
                test_gt = {}
                tag_len = self._test_taglens[idx]
                fixed_tag_len = self._max_tag_len + 2  # <start>...<end>
                if self._fixed_padding:
                    test_tags = torch.zeros(
                        1, fixed_tag_len, dtype=torch.long, device=self._device
                    )
                else:
                    test_tags = torch.zeros(
                        1, tag_len, dtype=torch.long, device=self._device
                    )
                test_tag_lens = torch.zeros(1, 1, dtype=torch.long).to(self._device)
                cells = []
                cell_lens = []
                cell_bboxes = []

        sample = self._image_fns[idx]
        # Rescale/convert the image and bboxes
        bboxes = self._bboxes[sample]
        sample_fn = self._get_image_path(sample)

        if not self._table_bboxes:
            table_bbox = None
        else:
            if sample in self._table_bboxes:
                table_bbox = self._table_bboxes[sample]
            else:
                table_bbox = None
        scaled_img, scaled_bboxes = self._transformer.sample_preprocessor(
            sample_fn, bboxes, self._purpose, table_bbox
        )

        imgs[0] = scaled_img.to(self._device)

        # Train specific output
        if self._purpose == s.TRAIN_PURPOSE:
            # Remove the padding from tags and cells
            if self._fixed_padding:
                tags[0] = torch.LongTensor(self._tags[idx]).to(self._device)
            else:
                tags[0] = torch.LongTensor(self._tags[idx][:tag_len]).to(self._device)

            tag_lens[0] = torch.LongTensor([self._taglens[idx]]).to(self._device)
            num_cells[0] = len(self._cell_lens[idx])

            if len(self._cell_lens[idx]) > 0:
                sample_max_cell_len = max(self._cell_lens[idx])
            else:
                sample_max_cell_len = 0

            if self._load_cells:
                image_trimmed_cells = [
                    self._cells[idx][x][0:sample_max_cell_len]
                    for x in range(0, len(self._cells[idx]))
                ]
                cells.append(torch.LongTensor(image_trimmed_cells).to(self._device))

            cell_lens.append(torch.LongTensor(self._cell_lens[idx]).to(self._device))
            if self._train_bbox:

                cell_bboxes.append(
                    [
                        torch.from_numpy(
                            np.array(scaled_bboxes, dtype=np.float32)[:, :4]
                        ).to(self._device),
                        torch.from_numpy(
                            np.array(scaled_bboxes, dtype=np.compat.long)[:, -1]
                        ).to(self._device),
                    ]
                )

        elif self._purpose == s.VAL_PURPOSE:
            # Remove the padding from tags and cells
            if self._fixed_padding:
                val_tags[0] = torch.LongTensor(self._val_tags[idx]).to(self._device)
            else:
                val_tags[0] = torch.LongTensor(self._val_tags[idx][:val_tag_len]).to(
                    self._device
                )

            val_tag_lens[0] = torch.LongTensor([self._val_taglens[idx]]).to(
                self._device
            )
            val_num_cells[0] = len(self._val_cell_lens[idx])

            if len(self._val_cell_lens[idx]) > 0:
                sample_max_cell_len = max(self._val_cell_lens[idx])
            else:
                sample_max_cell_len = 0

            if self._load_cells:
                val_image_trimmed_cells = [
                    self._val_cells[idx][x][0:sample_max_cell_len]
                    for x in range(0, len(self._cells[idx]))
                ]
                val_cells.append(
                    torch.LongTensor(val_image_trimmed_cells).to(self._device)
                )

            val_cell_lens.append(
                torch.LongTensor(self._val_cell_lens[idx]).to(self._device)
            )
            if self._train_bbox:
                val_cell_bboxes.append(
                    [
                        torch.from_numpy(
                            np.array(scaled_bboxes, dtype=np.float32)[:, :4]
                        ).to(self._device),
                        torch.from_numpy(
                            np.array(scaled_bboxes, dtype=np.compat.long)[:, -1]
                        ).to(self._device),
                    ]
                )
        # val specific output
        elif self._purpose == s.TEST_PURPOSE:
            if test_gt is not None:
                test_gt[sample] = self._test[sample]

                # Remove the padding from tags and cells
                if self._fixed_padding:
                    test_tags[0] = torch.LongTensor(self._test_tags[idx]).to(
                        self._device
                    )
                else:
                    test_tags[0] = torch.LongTensor(self._test_tags[idx][:tag_len]).to(
                        self._device
                    )
                test_tag_lens[0] = torch.LongTensor([self._test_taglens[idx]]).to(
                    self._device
                )
                if self._predict_bbox:
                    cell_bboxes.append(
                        [
                            torch.from_numpy(
                                np.array(scaled_bboxes, dtype=np.float32)[:, :4]
                            ).to(self._device),
                            torch.from_numpy(
                                np.array(scaled_bboxes, dtype=np.compat.long)[:, -1]
                            ).to(self._device),
                        ]
                    )

        output = {}

        # Samples is a list with the given image filename
        samples = [self._image_fns[idx]]
        # All data points presented in a dictionary, each wrapped into a list,
        # for easier batching and collating later on
        if self._purpose == s.TRAIN_PURPOSE:
            output["imgs"] = imgs
            output["tags"] = tags
            output["tag_lens"] = tag_lens
            output["num_cells"] = num_cells
            output["cells"] = cells
            output["cell_lens"] = cell_lens
            output["cell_bboxes"] = cell_bboxes
            output["samples"] = samples
            output["test_gt"] = test_gt
        elif self._purpose == s.VAL_PURPOSE:
            output["imgs"] = imgs
            output["tags"] = val_tags
            output["tag_lens"] = val_tag_lens
            output["num_cells"] = val_num_cells
            output["cells"] = val_cells
            output["cell_lens"] = val_cell_lens
            output["cell_bboxes"] = val_cell_bboxes
            output["samples"] = samples
            output["test_gt"] = test_gt
        elif self._purpose == s.TEST_PURPOSE:
            output["imgs"] = imgs
            output["tags"] = test_tags
            output["tag_lens"] = test_tag_lens
            output["num_cells"] = num_cells
            output["cells"] = cells
            output["cell_lens"] = cell_lens
            output["cell_bboxes"] = cell_bboxes
            output["samples"] = samples
            output["test_gt"] = test_gt
        else:
            output["imgs"] = imgs
            output["tags"] = tags
            output["tag_lens"] = tag_lens
            output["num_cells"] = num_cells
            output["cells"] = cells
            output["cell_lens"] = cell_lens
            output["cell_bboxes"] = cell_bboxes
            output["samples"] = samples
            output["test_gt"] = test_gt
        return output

    def get_batch_size(self):
        r"""
        Return the actual batch_size
        """
        return self._batch_size

    def reset(self):
        self._index = 0

    def __iter__(self):
        return self

    def is_valid(self, img, config):
        max_tag_len = config["preparation"]["max_tag_len"]
        max_cell_len = config["preparation"]["max_cell_len"]
        check_limits = True
        if "check_limits" in config["preparation"]:
            check_limits = config["preparation"]["check_limits"]
        if check_limits:
            if len(img["html"]["structure"]["tokens"]) > max_tag_len:
                self._log().debug(
                    "========================================= TAG LEN REJECTED"
                )
                self._log().debug("File name: {}".format(img["filename"]))
                tokens_len = len(img["html"]["structure"]["tokens"])
                self._log().debug("Structure token len: {}".format(tokens_len))
                self._log().debug("More than max_tag_len: {}".format(max_tag_len))
                self._log().debug(
                    "=========================================================="
                )
                return False
            for cell in img["html"]["cells"]:
                if len(cell["tokens"]) > max_cell_len:
                    self._log().debug(
                        "======================================== CELL LEN REJECTED"
                    )
                    self._log().debug("File name: {}".format(img["filename"]))
                    self._log().debug("Cell len: {}".format(len(cell["tokens"])))
                    self._log().debug("More than max_cell_len: {}".format(max_cell_len))
                    self._log().debug(
                        "=========================================================="
                    )
                    return False
        self.raw_data_dir = config["preparation"]["raw_data_dir"]
        with Image.open(
            os.path.join(self.raw_data_dir, img["split"], img["filename"])
        ) as im:
            max_image_size = config["preparation"]["max_image_size"]
            if im.width > max_image_size or im.height > max_image_size:
                # IMG SIZE REJECTED
                return False
        return True

    def __next__(self):
        r"""
        Get the next batch or raise the StopIteration

        In order to have the batch size fixed also in the last iteration, we wrap over the dataset
        and repeat some of the first elements.

        Depending on the "purpose" different returned objects can be None.

        Returns
        -------
        imgs : tensor (batch_size, image_channels, resized_image, resized_image)
            Batch with the rescaled images
        tags : The tags of the images. The object is one of:
            None : If purpose is not "train"
            (batch_size, max_tag_len + 2) : If purpose is "train" and "fixed_padding" is true.
                                            The +2 over the max_tag_len is for the <start> <stop>
            (batch_size, batch_max_tag_len) : If purpose is "train" and "fixed_padding" is false,
                                              where "batch_max_tag_len" is the max length of the
                                              tags in batch
        tag_lens : The real length of the tags per image in the batch. The object is one of:
            None : If purpose is not "train"
            (batch_size, 1) : If purpose is "train"
        num_cells : The number of cells per image in the batch. The object is one of:
            None : If purpose is not "train"
            (batch_size, 1) : If purpose is "train"
        cells : The cell tags for the images in the batch. The object is one of:
            None : If purpose is not "train"
            list with LongTensor: If purpose is "train"
        cell_lens : The length of the cell tags per image in the batch. The object is one of:
            None : If purpose is not "train"
            list with LongTensor: If purpose is "train"
        cell_bboxes : The transformed (rescaled, padded, etc) bboxes of the cells for all images in
                      batch. Each list is a bbox in the format [xc, yc, w, h] where xc, yc are the
                      coords of the center, w, h the width and height of the bbox and all are
                      normalized to the scaled size of the image. The object is one of:
            None : If purpose is not "train"
            list of lists: If purpose is "train"
        samples : list of string
            The filenames in the batch
        val_gt : The ground turth raw attributes for the validation split. The object is one of:
            None : If purpose is not "val"
            dictionary : If purpose is "val"
        """

        if self._index >= self._dataset_size:
            raise StopIteration()

        # Compute the next sample
        if (
            self._dataset_size - self._index >= self._batch_size
        ):  # Full batch_size sample
            step = self._batch_size
            sample_indices = self._ind[self._index : self._index + step]
        else:
            # skip last batch
            raise StopIteration()

        batch = []
        # Loop over the batch indices and collect items for the batch
        for i, idx in enumerate(sample_indices):
            item = self.__getitem__(idx)
            batch.append(item)
        # Collate batch
        output = self.bcol(batch)
        self._index += step

        return output

    def shuffle(self):
        r"""
        Shuffle the training images
        This takes place only in case of training, otherwise it just returns

        Output: True in case a shuffling took place, False otherwise
        """
        if self._purpose != s.TRAIN_PURPOSE:
            return False

        # image_fns_np = np.asarray(self._image_fns)
        # To get a deterministic random shuffle, we need to seed our random
        # with a deterministic seed (int)
        np.random.seed(42)
        # Then shuffle after seeding
        self._ind = np.random.permutation(self._dataset_size)
        self._index = 0
        return True

    def get_init_data(self):
        r"""
        Create a dictionary with all kind of initialization data necessary for all models.
        This data should not be served by the __next__ method.
        """
        init_data = {"word_map": self._word_map, "statistics": self._statistics}
        return init_data

    def _get_image_path(self, img_fn):
        r"""
        Get the full image path out of the image file name
        """
        if self._dataset_type == "TF_prepared":
            if self._purpose == s.TRAIN_PURPOSE:
                full_fn = os.path.join(self._raw_data_dir, "train", img_fn)
            elif self._purpose == s.VAL_PURPOSE:
                full_fn = os.path.join(self._raw_data_dir, "val", img_fn)
            elif self._purpose == s.TEST_PURPOSE:
                full_fn = os.path.join(self._raw_data_dir, "test", img_fn)
            else:
                full_fn = os.path.join(self._raw_data_dir, img_fn)

        if full_fn is None or not os.path.isfile(full_fn):
            self._log().error("File not found: " + full_fn)
            return None

        return full_fn

    def format_html(self, img):
        r"""
        Formats HTML code from tokenized annotation of img
        """
        tag_len = len(img["html"]["structure"]["tokens"])
        if self._load_cells:
            cell_len_max = max([len(c["tokens"]) for c in img["html"]["cells"]])
        else:
            cell_len_max = 0

        HTML = img["html"]["structure"]["tokens"].copy()
        to_insert = [i for i, tag in enumerate(HTML) if tag in ("<td>", ">")]

        if self._load_cells:
            for i, cell in zip(to_insert[::-1], img["html"]["cells"][::-1]):
                if cell:
                    cell = "".join(
                        [
                            escape(token) if len(token) == 1 else token
                            for token in cell["tokens"]
                        ]
                    )
                    HTML.insert(i + 1, cell)

        HTML = "<html><body><table>%s</table></body></html>" % "".join(HTML)
        root = html.fromstring(HTML)
        if self._predict_bbox:
            for td, cell in zip(root.iter("td"), img["html"]["cells"]):
                if "bbox" in cell:
                    bbox = cell["bbox"]
                    td.attrib["x"] = str(bbox[0])
                    td.attrib["y"] = str(bbox[1])
                    td.attrib["width"] = str(bbox[2] - bbox[0])
                    td.attrib["height"] = str(bbox[3] - bbox[1])
        HTML = html.tostring(root, encoding="utf-8").decode()
        return HTML, tag_len, cell_len_max

    def _build_predict_cache(self):
        r"""
        populate cache with image file names that need to be predicted
        """
        self._prepared_data_dir = c.safe_get_parameter(
            self._config, ["dataset", "prepared_data_dir"], required=False
        )
        self._data_name = c.safe_get_parameter(
            self._config, ["dataset", "name"], required=True
        )

        if self._prepared_data_dir is None:

            self._statistics = c.safe_get_parameter(
                self._config, ["dataset", "image_normalization"], required=True
            )

            self._word_map = c.safe_get_parameter(
                self._config, ["dataset_wordmap"], required=True
            )

        else:
            # Load statistics
            statistics_fn = c.get_prepared_data_filename("STATISTICS", self._data_name)
            with open(os.path.join(self._prepared_data_dir, statistics_fn), "r") as f:
                self._log().debug("Load statistics from: {}".format(statistics_fn))
                self._statistics = json.load(f)

            # Load word_map
            word_map_fn = c.get_prepared_data_filename("WORDMAP", self._data_name)
            with open(os.path.join(self._prepared_data_dir, word_map_fn), "r") as f:
                self._log().debug("Load WORDMAP from: {}".format(word_map_fn))
                self._word_map = json.load(f)

        # Get Image File Names for Prediction
        self._image_fns = []
        self._bboxes = {}
        self._bboxes_table = {}
        self._raw_data_dir = self._predict_dir
        self._table_bboxes = {}

        if self._predict_dir[-1] != "/":
            self._predict_dir += "/"
        for path in list(glob(self._predict_dir + "**/*.png", recursive=True)):
            filename = os.path.basename(path)
            self._image_fns.append(filename)
            self._log().info("Image found: {}".format(filename))
            self._bboxes[filename] = []

        # Get size of a dataset to predict
        self._dataset_size = len(self._image_fns)

        # Get the number of image channels
        self._log().info(
            "To test load... {}".format(self._predict_dir + self._image_fns[0])
        )
        img = u.load_image(self._predict_dir + self._image_fns[0])
        if img is None:
            msg = "Cannot load image"
            self._log().error(msg)
            raise Exception(msg)
        self._img_ch = img.shape[0]

    def _build_cache(self):
        r"""
        Cache with small data
        """
        all_bboxes = {}  # Keep original bboxes for all images
        table_bboxes = {}
        self._log().info("Building the cache...")
        self._raw_data_dir = c.safe_get_parameter(
            self._config, ["dataset", "raw_data_dir"], required=True
        )
        self._prepared_data_dir = c.safe_get_parameter(
            self._config, ["dataset", "prepared_data_dir"], required=False
        )

        self._data_name = c.safe_get_parameter(
            self._config, ["dataset", "name"], required=True
        )

        if self._prepared_data_dir is None:

            self._statistics = c.safe_get_parameter(
                self._config, ["dataset", "image_normalization"], required=True
            )

            self._word_map = c.safe_get_parameter(
                self._config, ["dataset_wordmap"], required=True
            )

        else:
            # Load statistics
            statistics_fn = c.get_prepared_data_filename("STATISTICS", self._data_name)
            with open(os.path.join(self._prepared_data_dir, statistics_fn), "r") as f:
                self._log().debug("Load statistics from: {}".format(statistics_fn))
                self._statistics = json.load(f)

            # Load word_map
            word_map_fn = c.get_prepared_data_filename("WORDMAP", self._data_name)
            with open(os.path.join(self._prepared_data_dir, word_map_fn), "r") as f:
                self._log().debug("Load WORDMAP from: {}".format(word_map_fn))
                self._word_map = json.load(f)

        word_map_cell = self._word_map["word_map_cell"]
        word_map_tag = self._word_map["word_map_tag"]
        # Read image paths and captions for each image
        train_image_paths = []
        train_images = []
        val_image_paths = []
        val_images = []
        test_image_paths = []
        test_images = []
        predict_images = []

        train_image_tags = (
            []
        )  # List of list of structure tokens for each image in the train set
        train_image_cells = []
        train_image_cells_len = []

        val_image_tags = (
            []
        )  # List of list of structure tokens for each image in the train set
        val_image_cells = []
        val_image_cells_len = []

        test_image_tags = []
        test_gt = dict()

        invalid_files = 0
        total_files = 0

        self._log().debug(
            "Create lists with image filenames per split, train tags/cells and GT"
        )
        with jsonlines.open(self.annotation, "r") as reader:
            for img in tqdm(reader):
                total_files += 1
                img_filename = img["filename"]
                path = os.path.join(self._raw_data_dir, img["split"], img_filename)

                # Keep bboxes for all images
                all_cell_bboxes = []
                for cell in img["html"]["cells"]:
                    if "bbox" not in cell:
                        continue
                    all_cell_bboxes.append(cell["bbox"])
                all_bboxes[img_filename] = all_cell_bboxes

                # if dataset does include bbox for the table itself
                if "table_bbox" in img:
                    table_bboxes[img_filename] = img["table_bbox"]
                if img["split"] == "train":
                    if self._purpose == s.TRAIN_PURPOSE:
                        # Skip invalid images
                        if not self.is_valid(img, self._config):
                            invalid_files += 1
                            continue
                        tags = []
                        cells = []
                        cell_lens = []
                        tags.append(img["html"]["structure"]["tokens"])

                        if self._load_cells:
                            for cell in img["html"]["cells"]:
                                cells.append(cell["tokens"])
                                cell_lens.append(len(cell["tokens"]) + 2)
                        else:
                            for cell in img["html"]["cells"]:
                                cell_lens.append(len(cell["tokens"]) + 2)

                        train_images.append(img_filename)
                        train_image_paths.append(path)
                        train_image_tags.append(tags)
                        train_image_cells.append(cells)
                        train_image_cells_len.append(cell_lens)
                if img["split"] == "val":
                    if self._purpose == s.VAL_PURPOSE:
                        # Skip invalid images
                        if not self.is_valid(img, self._config):
                            invalid_files += 1
                            continue

                        val_tags = []
                        val_cells = []
                        val_cell_lens = []
                        val_tags.append(img["html"]["structure"]["tokens"])

                        if self._load_cells:
                            for cell in img["html"]["cells"]:
                                val_cells.append(cell["tokens"])
                                val_cell_lens.append(len(cell["tokens"]) + 2)
                        else:
                            for cell in img["html"]["cells"]:
                                val_cell_lens.append(len(cell["tokens"]) + 2)

                        with Image.open(path) as im:
                            HTML, tag_len, cell_len_max = self.format_html(img)
                            lt1 = [">", "lcel", "ucel", "xcel"]
                            lt2 = img["html"]["structure"]["tokens"]
                            tcheck = any(item in lt1 for item in lt2)
                            if tcheck:
                                gtt = "complex"
                            else:
                                gtt = "simple"
                            test_gt[img_filename] = {
                                "html": HTML,
                                "tag_len": tag_len,
                                "cell_len_max": cell_len_max,
                                "width": im.width,
                                "height": im.height,
                                "type": gtt,
                                "html_tags": img["html"]["structure"]["tokens"],
                            }

                        val_images.append(img_filename)
                        val_image_paths.append(path)
                        val_image_tags.append(val_tags)
                        val_image_cells.append(val_cells)
                        val_image_cells_len.append(val_cell_lens)

                elif img["split"] == "test":
                    if self._purpose == s.TEST_PURPOSE:
                        # Skip invalid images
                        if not self.is_valid(img, self._config):
                            invalid_files += 1
                            continue

                        with Image.open(path) as im:
                            HTML, tag_len, cell_len_max = self.format_html(img)
                            lt1 = [">", "lcel", "ucel", "xcel"]
                            lt2 = img["html"]["structure"]["tokens"]
                            tcheck = any(item in lt1 for item in lt2)
                            if tcheck:
                                gtt = "complex"
                            else:
                                gtt = "simple"
                            test_gt[img_filename] = {
                                "html": HTML,
                                "tag_len": tag_len,
                                "cell_len_max": cell_len_max,
                                "width": im.width,
                                "height": im.height,
                                "type": gtt,
                                "html_tags": img["html"]["structure"]["tokens"],
                            }
                        test_images.append(img_filename)

                        test_tags = []
                        test_tags.append(img["html"]["structure"]["tokens"])
                        test_image_paths.append(path)
                        test_image_tags.append(test_tags)
                else:
                    if self._purpose == s.PREDICT_PURPOSE:
                        predict_images.append(img_filename)

        image_fns = {
            s.TRAIN_SPLIT: train_images,
            s.VAL_SPLIT: val_images,
            s.TEST_SPLIT: test_images,
        }

        self._log().debug("Keep the split data pointed by the purpose")
        # Images
        # Filter out the images for the particual split
        self._image_fns = image_fns[self._purpose]
        self._dataset_size = len(self._image_fns)
        assert len(self._image_fns) > 0, "Empty image split: " + self._purpose

        # Get the number of image channels
        img = u.load_image(self._get_image_path(self._image_fns[0]))
        if img is None:
            msg = "Cannot load image"
            self._log().error(msg)
            raise Exception(msg)
        self._img_ch = img.shape[0]

        # img_name -> list of bboxes, each bbox is a list with x1y1x2y2
        split_bboxes = {}
        img_names = set(self._image_fns)  # Set will speed up search
        for img_name, bbox in all_bboxes.items():
            if img_name not in img_names:
                continue
            if img_name not in split_bboxes:
                split_bboxes[img_name] = []
            # we should use extend not append, otherwise we get list within list
            split_bboxes[img_name].extend(bbox)
        self._bboxes = split_bboxes
        self._table_bboxes = table_bboxes
        # -------------------------------------------------------------------------------
        # Train specific
        # -------------------------------------------------------------------------------
        # Compute encoded tags and cells
        enc_tags = []
        tag_lens = []
        enc_cells = []
        cell_lens = []

        val_enc_tags = []
        val_tag_lens = []
        val_enc_cells = []
        val_cell_lens = []

        test_enc_tags = []
        test_tag_lens = []

        # Based on the "purpose"
        if self._purpose == s.TRAIN_PURPOSE:
            self._log().debug("Convert train tags and cell tags to indices")
            for i, path in enumerate(tqdm(train_image_paths)):
                for tag in train_image_tags[i]:
                    # Encode tags
                    # Notice that at this point we don't have images longer than max_tag_length
                    # The same happens with the cell tokens
                    enc_tag = (
                        [word_map_tag["<start>"]]
                        + [
                            word_map_tag.get(word, word_map_tag["<unk>"])
                            for word in tag
                        ]
                        + [word_map_tag["<end>"]]
                        + [word_map_tag["<pad>"]] * (self._max_tag_len - len(tag))
                    )
                    # Find caption lengths
                    tag_len = len(tag) + 2

                    enc_tags.append(enc_tag)
                    tag_lens.append(tag_len)

                enc_cell_seq = []
                cell_seq_len = []

                if self._load_cells:
                    for cell in train_image_cells[i]:
                        # Encode captions
                        enc_cell = (
                            [word_map_cell["<start>"]]
                            + [
                                word_map_cell.get(word, word_map_cell["<unk>"])
                                for word in cell
                            ]
                            + [word_map_cell["<end>"]]
                            + [word_map_cell["<pad>"]]
                            * (self._max_cell_len - len(cell))
                        )
                        enc_cell_seq.append(enc_cell)
                        # Find caption lengths
                        cell_len = len(cell) + 2
                        cell_seq_len.append(cell_len)
                else:
                    for cell in train_image_cells_len[i]:
                        cell_seq_len.append(cell)
                enc_cells.append(enc_cell_seq)
                cell_lens.append(cell_seq_len)

        if self._purpose == s.VAL_PURPOSE:
            self._log().debug("Convert train tags and cell tags to indices")
            for i, path in enumerate(tqdm(val_image_paths)):
                for tag in val_image_tags[i]:
                    # Encode tags
                    # Notice that at this point we don't have images longer than max_tag_length
                    # The same happens with the cell tokens
                    val_enc_tag = (
                        [word_map_tag["<start>"]]
                        + [
                            word_map_tag.get(word, word_map_tag["<unk>"])
                            for word in tag
                        ]
                        + [word_map_tag["<end>"]]
                        + [word_map_tag["<pad>"]] * (self._max_tag_len - len(tag))
                    )
                    # Find caption lengths
                    val_tag_len = len(tag) + 2

                    val_enc_tags.append(val_enc_tag)
                    val_tag_lens.append(val_tag_len)

                val_enc_cell_seq = []
                val_cell_seq_len = []

                if self._load_cells:
                    for cell in val_image_cells[i]:
                        # Encode captions
                        val_enc_cell = (
                            [word_map_cell["<start>"]]
                            + [
                                word_map_cell.get(word, word_map_cell["<unk>"])
                                for word in cell
                            ]
                            + [word_map_cell["<end>"]]
                            + [word_map_cell["<pad>"]]
                            * (self._max_cell_len - len(cell))
                        )
                        val_enc_cell_seq.append(val_enc_cell)

                    # Find caption lengths
                    cell_len = len(cell) + 2
                    val_cell_seq_len.append(cell_len)
                else:
                    for cell in val_image_cells_len[i]:
                        val_cell_seq_len.append(cell)
                val_enc_cells.append(val_enc_cell_seq)
                val_cell_lens.append(val_cell_seq_len)

        if self._purpose == s.TEST_PURPOSE:
            self._log().debug("Convert val tags to indices")
            for i, path in enumerate(tqdm(test_image_paths)):
                for tag in test_image_tags[i]:
                    # Encode tags
                    # Notice that at this point we don't have images longer than max_tag_length
                    # The same happens with the cell tokens
                    test_enc_tag = (
                        [word_map_tag["<start>"]]
                        + [
                            word_map_tag.get(word, word_map_tag["<unk>"])
                            for word in tag
                        ]
                        + [word_map_tag["<end>"]]
                        + [word_map_tag["<pad>"]] * (self._max_tag_len - len(tag))
                    )

                    # Find caption lengths
                    test_tag_len = len(tag) + 2

                    test_enc_tags.append(test_enc_tag)
                    test_tag_lens.append(test_tag_len)

        self._tags = enc_tags
        self._taglens = tag_lens
        self._cells = enc_cells
        self._cell_lens = cell_lens

        # -------------------------------------------------------------------------------
        # val specific
        # -------------------------------------------------------------------------------
        self._val_tags = val_enc_tags
        self._val_taglens = val_tag_lens
        self._val_cells = val_enc_cells
        self._val_cell_lens = val_cell_lens
        # -------------------------------------------------------------------------------
        # test / evaluation specific
        # -------------------------------------------------------------------------------
        self._test = test_gt
        self._test_tags = test_enc_tags
        self._test_taglens = test_tag_lens
