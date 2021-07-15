from typing import Tuple

from panda3d.core import Vec3, NodePath
from pgdrive.constants import CamMask
from pgdrive.engine.world.image_buffer import ImageBuffer


class MiniMap(ImageBuffer):
    CAM_MASK = CamMask.MiniMap
    default_region = [0., 1 / 3, ImageBuffer.display_bottom, ImageBuffer.display_top]

    def __init__(self, para: Tuple, chassis_np: NodePath):
        self.BUFFER_W = para[0]
        self.BUFFER_H = para[1]
        height = para[2]
        super(MiniMap, self).__init__(
            self.BUFFER_W, self.BUFFER_H, Vec3(0, 20, height), self.BKG_COLOR, parent_node=chassis_np
        )
        self.cam.lookAt(Vec3(0, 20, 0))
        self.lens.setAspectRatio(2.0)
        self.add_to_display(self.default_region)
