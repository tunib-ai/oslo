#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
from oslo.constants import IS_TENSOR_PARALLEL, NUM_PARTITIONS


def divide(numerator, denominator):
    """Only allow exact division.

    Args:
        numerator (int): Numerator of the division.
        denominator (int): Denominator of the division.

    Returns:
        int: the result of exact division.
    """
    assert denominator != 0, 'denominator can not be zero'
    assert numerator % denominator == 0, \
        '{} is not divisible by {}'.format(numerator, denominator)
    return numerator // denominator


def set_tensor_parallel_attribute_by_size(param, size):
    setattr(param, IS_TENSOR_PARALLEL, True)
    setattr(param, NUM_PARTITIONS, size // np.prod(param.shape))


def set_tensor_parallel_attribute_by_partition(param, num_partitions):
    setattr(param, IS_TENSOR_PARALLEL, True)
    setattr(param, NUM_PARTITIONS, num_partitions)
