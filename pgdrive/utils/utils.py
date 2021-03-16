import copy
import logging
import os
import sys
import uuid

from pgdrive.pg_config import PGConfig


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


def random_string(prefix=None):
    ret = str(uuid.uuid4())
    if prefix is not None:
        ret = "{}-{}".format(prefix, ret)
    return ret


# The following two functions is copied from ray/tune/utils/util.py, raise_error and pgconfig support is added by us!
def merge_dicts(old_dict, new_dict, raise_error=True):
    """
    Args:
        old_dict (dict): Dict 1.
        new_dict (dict): Dict 2.
        raise_error (bool): Whether to raise error if new key is found.

    Returns:
         dict: A new dict that is d1 and d2 deep merged.
    """
    if isinstance(old_dict, PGConfig):
        old_dict = old_dict.get_dict()
    if isinstance(new_dict, PGConfig):
        new_dict = new_dict.get_dict()
    merged = copy.deepcopy(old_dict)
    deep_update(merged, new_dict, new_keys_allowed=False, allow_new_subkey_list=[], raise_error=raise_error)
    return PGConfig(merged)


def deep_update(
    original,
    new_dict,
    new_keys_allowed=False,
    allow_new_subkey_list=None,
    override_all_if_type_changes=None,
    raise_error=True
):
    allow_new_subkey_list = allow_new_subkey_list or []
    override_all_if_type_changes = override_all_if_type_changes or []

    for k, value in new_dict.items():
        if k not in original and not new_keys_allowed:
            if raise_error:
                raise Exception("Unknown config parameter `{}` ".format(k))
            else:
                continue

        # Both orginal value and new one are dicts.
        if isinstance(original.get(k), dict) and isinstance(value, dict):
            # Check old type vs old one. If different, override entire value.
            if k in override_all_if_type_changes and \
                    "type" in value and "type" in original[k] and \
                    value["type"] != original[k]["type"]:
                original[k] = value
            # Allowed key -> ok to add new subkeys.
            elif k in allow_new_subkey_list:
                deep_update(original[k], value, True, raise_error=raise_error)
            # Non-allowed key.
            else:
                deep_update(original[k], value, new_keys_allowed, raise_error=raise_error)
        # Original value not a dict OR new value not a dict:
        # Override entire value.
        else:
            original[k] = value
    return original
