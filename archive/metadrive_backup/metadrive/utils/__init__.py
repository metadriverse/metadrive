from metadrive.utils.config import Config, merge_config_with_unknown_keys, merge_config
from metadrive.utils.coordinates_shift import panda_heading, panda_vector, metadrive_heading, metadrive_vector
from metadrive.utils.math import safe_clip, clip, norm, distance_greater, safe_clip_for_small_array, Vector
from metadrive.utils.random_utils import get_np_random, random_string
from metadrive.utils.registry import get_metadrive_class
from metadrive.utils.utils import is_mac, import_pygame, recursive_equal, setup_logger, merge_dicts, \
    concat_step_infos, is_win, time_me
