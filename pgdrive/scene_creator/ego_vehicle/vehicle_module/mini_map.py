from typing import Tuple

from panda3d.core import Vec3, NodePath
from pgdrive.pg_config.cam_mask import CamMask
from pgdrive.world.image_buffer import ImageBuffer


class MiniMap(ImageBuffer):
    CAM_MASK = CamMask.MiniMap

    def __init__(self, para: Tuple, chassis_np: NodePath, pg_world):
        self.BUFFER_W = para[0]
        self.BUFFER_H = para[1]
        height = para[2]
        super(MiniMap, self).__init__(
            self.BUFFER_W,
            self.BUFFER_H,
            Vec3(0, 20, height),
            self.BKG_COLOR,
            pg_world=pg_world,
            parent_node=chassis_np
        )
        self.cam.lookAt(Vec3(0, 20, 0))
        self.lens.setAspectRatio(2.0)
        self.add_to_display(pg_world, [0., 1 / 3, self.display_bottom, self.display_top])
