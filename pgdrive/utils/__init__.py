import logging
import os
import sys

from pgdrive.utils.asset_loader import AssetLoader, initialize_asset_loader
from pgdrive.utils.math_utils import safe_clip, clip, norm
from pgdrive.utils.random import get_np_random, RandomEngine


def import_pygame():
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    import pygame
    return pygame


def setup_logger(debug=False):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    )


def recursive_equal(data1, data2, need_assert=False):
    if isinstance(data1, dict):
        is_ins = isinstance(data2, dict)
        key_right = set(data1.keys()) == set(data2.keys())
        if need_assert:
            assert is_ins and key_right, (data1.keys(), data2.keys())
        if not (is_ins and key_right):
            return False
        ret = []
        for k in data1:
            ret.append(recursive_equal(data1[k], data2[k]))
        return all(ret)

    elif isinstance(data1, list):
        len_right = len(data1) == len(data2)
        is_ins = isinstance(data2, list)
        if need_assert:
            assert len_right and is_ins, (len(data1), len(data2), data1, data2)
        if not (is_ins and len_right):
            return False
        ret = []
        for i in range(len(data1)):
            ret.append(recursive_equal(data1[i], data2[i]))
        return all(ret)

    else:
        ret = data1 == data2
        if need_assert:
            assert ret, (type(data1), type(data2), data1, data2)
        return ret


def is_mac():
    return sys.platform == "darwin"
