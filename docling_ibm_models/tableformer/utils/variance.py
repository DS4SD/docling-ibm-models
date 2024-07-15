#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import logging

import numpy as np

import docling_ibm_models.tableformer.settings as s

LOG_LEVEL = logging.INFO


class MyWelford:
    r"""
    Running computation of the sample mean and sample variance using Welford's algorithm
    """

    def __init__(self):
        self._i = 0  # Running index
        self._m = 0  # Running mean
        self._s = 0  # (n - 1) * variance

    def reset(self):
        r"""
        Reset the object
        """
        self._i = 0
        self._m = 0
        self._s = 0

    def add(self, xi):
        r"""
        Invoke add each time a new sample arrives

        Inputs:
           xi: The next sample of data
        """
        self._i += 1
        old_m = self._m
        self._m = self._m + (xi - self._m) / self._i
        self._s = self._s + (xi - self._m) * (xi - old_m)

    def results(self):
        r"""
        Get the computed mean, variance and standard deviation up to now

        Outputs:
            m: Sample mean
            v: Sample variance
            std: Sample standard deviation
        """
        if self._i <= 1:
            return None, None, None

        # v = self._s / (self._i - 1)  # Sample variance
        v = self._s / (self._i)  # Population variance
        std = np.sqrt(v)
        return self._m, v, std


class MyWelfordImg(MyWelford):
    r"""
    Welford algorithm to calculate running mean and sample variance for images
    """

    def __init__(self):
        super(MyWelfordImg, self).__init__()

    def add(self, img):
        r"""
        Input:
            img: An image numpy array (channel, width, height). The only requirement is to have the
                 channels as the first dimension and have 3 dimensions in total
        """
        channels = img.shape[0]
        flat_dim = img.shape[1] * img.shape[2]
        img_r = img.reshape(channels, flat_dim)

        for i in range(flat_dim):
            super(MyWelfordImg, self).add(img_r[:, i])


class ChanVarianceImg:
    r"""
    Chan's algorithm to compute a running variance with support of sub-samples
    In this implementation each sub-sample is an images

    Math for the original paper:
    https://github.ibm.com/nli/variance_formulae
    """

    def __init__(self):
        r""" """
        self._first = True
        # Size of the calculated dataset
        self._n = 0
        # Sum of the samples for the 3 image channels
        self._t = 0
        # Sum of the square differences of the deviations of the samples from the mean
        self._s = 0

    def add(self, img):
        r"""
        Add the provided image to the computation of the dataset statistics

        Input:
            img: An image numpy array (channel, width, height). The only requirement is to have the
                 channels as the first dimension and have 3 dimensions in total
        """
        ch = img.shape[0]
        n = img.shape[1] * img.shape[2]
        img = img.reshape(ch, n)
        img_t = img.sum(axis=1)
        img_t_v = img_t.reshape(ch, 1)
        diff = (img - (img_t_v / n)) ** 2
        img_s = diff.sum(axis=1)

        if not self._first:
            c = (self._n / (n * (self._n + n))) * (
                ((n / self._n) * self._t - img_t) ** 2
            )
            self._s += img_s + c
            self._t += img_t
        else:
            self._s = img_s
            self._t = img_t
            self._first = False
        self._n += n

    def results(self):
        r"""
        Get the computed statistics

        Output:
            mean: Mean for the complete dataset
            var: Population variance for the complete dataset
            std: Population standard deviation for the complete dataset
        """
        mean = list(self._t / self._n)
        var = list(self._s / self._n)  # Population variance
        std = list(np.sqrt(var))

        return mean, var, std

    def reset(self):
        r"""
        Reset the object to start over again
        """
        self._n = 0
        self._t = 0
        self._s = 0
        self._first = True


if __name__ == "__main__":
    logger = s.get_custom_logger("variance", LOG_LEVEL)

    n = 50000
    channels = 3
    width = 448
    height = 448

    my = ChanVarianceImg()
    # Generate random images
    for i in range(n):
        logger.info(i)
        img = 255 * np.random.rand(channels, width, height)
        my.add(img)

    # Calculate the statistics
    m, v, std = my.results()
    assert m.shape == (3,), "Wrong mean dimension"
    assert v.shape == (3,), "Wrong variance dimension"
    assert std.shape == (3,), "Wrong std dimension"
