from metadrive.utils.config import Config, merge_config_with_unknown_keys, merge_config
from metadrive.utils.coordinates_shift import panda_heading, panda_vector, metadrive_heading, metadrive_vector
from metadrive.utils.math import safe_clip, clip, norm, distance_greater, safe_clip_for_small_array, Vector
from metadrive.utils.random_utils import get_np_random, random_string
from metadrive.utils.registry import get_metadrive_class
from metadrive.utils.utils import is_mac, import_pygame, recursive_equal, setup_logger, merge_dicts, \
    concat_step_infos, is_win
import pygame
from PIL import Image
from textwrap import dedent
from inspect import getsource
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter


def generate_gif(frames, gif_name="demo.gif", is_pygame_surface=False, duration=30):
    """

    Args:
        frames: a list of images or pygame surfaces
        gif_name: name of the file
        is_pygame_surface: convert pygame surface to PIL.image
        duration: controlling the duration of each frame, unit: ms

    Returns:

    """
    assert gif_name.endswith("gif"), "File name should end with .gif"
    imgs = [pygame.surfarray.array3d(frame) if is_pygame_surface else frame for frame in frames]
    imgs = [Image.fromarray(img) for img in imgs]
    imgs[0].save(gif_name, save_all=True, append_images=imgs[1:], duration=duration, loop=0)


def print_source(x):
    """
    Print the source code of module x
    Args:
        x:

    Returns:

    """
    code = dedent(getsource(x))
    print(highlight(code, PythonLexer(), TerminalFormatter()))
