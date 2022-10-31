import copy
import logging
import os
import sys
import datetime

from panda3d.bullet import BulletBodyNode


def import_pygame():
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    import pygame
    return pygame


def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def setup_logger(debug=False):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    )


def recursive_equal(data1, data2, need_assert=False):
    from metadrive.utils.config import Config
    if isinstance(data1, Config):
        data1 = data1.get_dict()
    if isinstance(data2, Config):
        data2 = data2.get_dict()

    if isinstance(data1, dict):
        is_ins = isinstance(data2, dict)
        key_right = set(data1.keys()) == set(data2.keys())
        if need_assert:
            assert is_ins and key_right, (data1.keys(), data2.keys())
        if not (is_ins and key_right):
            return False
        ret = []
        for k in data1:
            ret.append(recursive_equal(data1[k], data2[k], need_assert=need_assert))
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
            ret.append(recursive_equal(data1[i], data2[i], need_assert=need_assert))
        return all(ret)

    else:
        ret = data1 == data2
        if need_assert:
            assert ret, (type(data1), type(data2), data1, data2)
        return ret


def is_mac():
    return sys.platform == "darwin"


def is_win():
    return sys.platform == "win32"


def concat_step_infos(step_info_list):
    """We only conduct simply shallow update here!"""
    old_dict = dict()
    for new_dict in step_info_list:
        old_dict = merge_dicts(old_dict, new_dict, allow_new_keys=True, without_copy=True)
    return old_dict


# The following two functions is copied from ray/tune/utils/util.py, raise_error and pgconfig support is added by us!
def merge_dicts(old_dict, new_dict, allow_new_keys=False, without_copy=False):
    """
    Args:
        old_dict (dict, Config): Dict 1.
        new_dict (dict, Config): Dict 2.
        raise_error (bool): Whether to raise error if new key is found.

    Returns:
         dict: A new dict that is d1 and d2 deep merged.
    """
    old_dict = old_dict or dict()
    new_dict = new_dict or dict()
    if without_copy:
        merged = old_dict
    else:
        merged = copy.deepcopy(old_dict)
    _deep_update(
        merged, new_dict, new_keys_allowed=allow_new_keys, allow_new_subkey_list=[], raise_error=not allow_new_keys
    )
    return merged


def _deep_update(
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
                _deep_update(original[k], value, True, raise_error=raise_error)
            # Non-allowed key.
            else:
                _deep_update(original[k], value, new_keys_allowed, raise_error=raise_error)
        # Original value not a dict OR new value not a dict:
        # Override entire value.
        else:
            original[k] = value
    return original


def deprecation_warning(old, new, error=False) -> None:
    """Warns (via the `logger` object) or throws a deprecation warning/error.

    Args:
        old (str): A description of the "thing" that is to be deprecated.
        new (Optional[str]): A description of the new "thing" that replaces it.
        error (Optional[Union[bool, Exception]]): Whether or which exception to
            throw. If True, throw ValueError. If False, just warn.
            If Exception, throw that Exception.
    """
    msg = "`{}` has been deprecated.{}".format(old, (" Use `{}` instead.".format(new) if new else ""))
    if error is True:
        raise ValueError(msg)
    elif error and issubclass(error, Exception):
        raise error(msg)
    else:
        logger = logging.getLogger(__name__)
        logger.warning("DeprecationWarning: " + msg + " This will raise an error in the future!")


def get_object_from_node(node: BulletBodyNode):
    """
    Use this api to get the python object from bullet RayCast/SweepTest/CollisionCallback result
    """
    if node.getPythonTag(node.getName()) is None:
        return None
    from metadrive.engine.engine_utils import get_object
    ret = node.getPythonTag(node.getName()).base_object_name
    if isinstance(ret, str):
        return get_object(ret)[ret]
    else:
        return ret
