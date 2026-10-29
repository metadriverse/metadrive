from typing import Optional, Union, Iterable

import numpy as np
import cv2
from metadrive.engine.top_down_renderer import draw_top_down_map_native as native_draw
from metadrive.utils.utils import import_pygame

pygame = import_pygame()


def draw_top_down_map(map,
                      resolution: Iterable = (512, 512),
                      semantic_map=True) -> Optional[Union[np.ndarray, pygame.Surface]]:
    ret = native_draw(map, return_surface=False, semantic_map=semantic_map)
    return cv2.resize(ret, resolution, interpolation=cv2.INTER_LINEAR)
