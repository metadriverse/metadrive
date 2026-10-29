import copy
import datetime
import logging
import os
import sys
import time
import socket
import numpy as np
from panda3d.bullet import BulletBodyNode

from metadrive.constants import MetaDriveType


def is_port_occupied(port, host='127.0.0.1'):
    """
    Check if a given port is occupied on the specified host.

    :param port: Port number to check.
    :param host: Host address to check the port on. Default is '127.0.0.1'.
    :return: True if the port is occupied, False otherwise.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex((host, port))
        return result == 0


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

    if isinstance(data1, np.ndarray):
        tmp = np.asarray(data2)
        return np.all(data1 == tmp)

    if isinstance(data2, np.ndarray):
        tmp = np.asarray(data1)
        return np.all(tmp == data2)

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

    elif isinstance(data1, (list, tuple)):
        len_right = len(data1) == len(data2)
        is_ins = isinstance(data2, (list, tuple))
        if need_assert:
            assert len_right and is_ins, (len(data1), len(data2), data1, data2)
        if not (is_ins and len_right):
            return False
        ret = []
        for i in range(len(data1)):
            ret.append(recursive_equal(data1[i], data2[i], need_assert=need_assert))
        return all(ret)
    elif isinstance(data1, np.ndarray):
        ret = np.isclose(data1, data2).all()
        if need_assert:
            assert ret, (type(data1), type(data2), data1, data2)
        return ret
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
    from metadrive.engine.engine_utils import get_engine
    ret = node.getPythonTag(node.getName()).base_object_name
    is_road = MetaDriveType.is_lane(node.getPythonTag(node.getName()).type_name)
    if is_road:
        return get_engine().current_map.road_network.get_lane(ret)
    else:
        return get_object(ret)[ret]


def is_map_related_instance(obj):
    from metadrive.component.block.base_block import BaseBlock
    from metadrive.component.map.base_map import BaseMap
    return True if isinstance(obj, BaseBlock) or isinstance(obj, BaseMap) else False


def is_map_related_class(object_class):
    from metadrive.component.block.base_block import BaseBlock
    from metadrive.component.map.base_map import BaseMap
    return True if issubclass(object_class, BaseBlock) or issubclass(object_class, BaseMap) else False


def dict_recursive_remove_array(d):
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, dict):
        for k in d.keys():
            d[k] = dict_recursive_remove_array(d[k])
    return d


def time_me(fn):
    """
    Wrapper for testing the function time
    Args:
        fn: function

    Returns: None

    """
    def _wrapper(*args, **kwargs):
        start = time.time()
        ret = fn(*args, **kwargs)
        print("function: %s cost %s second" % (fn.__name__, time.time() - start))
        return ret

    return _wrapper


def time_me_with_prefix(prefix):
    """
    Wrapper for testing the function time
    Args:
        prefix: add a string to the function name itself

    Returns: None

    """
    def decorator(fn):
        def _wrapper(*args, **kwargs):
            start = time.time()
            ret = fn(*args, **kwargs)
            print(prefix, "function: %s cost %s second" % (fn.__name__, time.time() - start))
            return ret

        return _wrapper

    return decorator


def create_rectangle_from_midpoints(p1, p2, width, length_factor=1.0):
    """
    Create the vertices of a rectangle given two midpoints on opposite sides, the width of the rectangle,
    and an optional factor to scale the length of the rectangle.

    This function calculates the four vertices of a rectangle in 2D space. The rectangle's length is the distance
    between two provided midpoints (p1 and p2) scaled by the 'length_factor', and its width is specified by the
    'width' parameter. The rectangle is aligned with the line segment connecting p1 and p2.

    Parameters:
    p1 (list or tuple): The first midpoint on the rectangle's width edge (x, y).
    p2 (list or tuple): The second midpoint on the rectangle's width edge (x, y).
    width (float): The width of the rectangle.
    length_factor (float, optional): The factor by which to scale the length of the rectangle. Default is 1.0.

    Returns:
    numpy.ndarray: A 2D array containing four vertices of the rectangle, in the order
                   [bottom_left, top_left, top_right, bottom_right].

    Example:
    p1 = [1, 1]
    p2 = [4, 4]
    width = 2
    length_factor = 1.5
    create_rectangle_from_midpoints(p1, p2, width, length_factor)
    array([[ some array ]])
    """
    # Calculate the vector from point 1 to point 2 and its length
    v = np.array(p2) - np.array(p1)

    # Scale the vector by length_factor
    v_scaled = v * length_factor

    # Calculate the scaled midpoints
    midpoint = (np.array(p1) + np.array(p2)) / 2
    p1_scaled = midpoint - v_scaled / 2
    p2_scaled = midpoint + v_scaled / 2

    # Normalize the vector
    v_norm = v_scaled / np.linalg.norm(v_scaled)

    # Rotate 90 degrees to get the perpendicular vector
    perp_v = np.array([-v_norm[1], v_norm[0]])

    # Calculate the half width
    half_width = width / 2

    # Calculate the 4 corners of the rectangle
    p3 = p1_scaled + perp_v * half_width
    p4 = p1_scaled - perp_v * half_width
    p5 = p2_scaled + perp_v * half_width
    p6 = p2_scaled - perp_v * half_width

    # Return the 4 corners in order
    return np.array([p4, p3, p5, p6])


def draw_polygon(polygon):
    """
    Visualize the polygon with matplot lib
    Args:
        polygon: a list of 2D points

    Returns: None

    """
    import matplotlib.pyplot as plt

    # Create the rectangle
    rectangle_points = np.array(polygon)
    # Extract the points for easier plotting
    x_rect, y_rect = rectangle_points.T

    # Extract the original midpoints

    # Plot the rectangle
    plt.figure(figsize=(8, 8))
    plt.plot(*zip(*np.append(rectangle_points, [rectangle_points[0]], axis=0)), marker='o', label='Rectangle Vertices')
    plt.fill(
        *zip(*np.append(rectangle_points, [rectangle_points[0]], axis=0)), alpha=0.3
    )  # Fill the rectangle with light opacity

    # Plot the original midpoints
    # plt.scatter(x_mid, y_mid, color='red', zorder=5, label='Midpoints')

    # Set equal scaling and labels
    plt.axis('equal')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Visualization of the Rectangle and Input Points')
    plt.legend()
    plt.grid(True)
    plt.show()
