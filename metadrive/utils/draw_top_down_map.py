from typing import Optional, Union, Iterable

import numpy as np

from metadrive.obs.top_down_renderer import draw_top_down_map as native_draw
from metadrive.utils.utils import import_pygame

pygame, gfxdraw = import_pygame()


def draw_top_down_map(map, resolution: Iterable = (512, 512), semantic_map=True) -> Optional[Union[np.ndarray, pygame.Surface]]:
    return native_draw(map, resolution, return_surface=False, semantic_map=semantic_map)
