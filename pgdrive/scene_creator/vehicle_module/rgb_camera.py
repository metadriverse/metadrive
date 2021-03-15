from panda3d.core import Vec3, NodePath

from pgdrive.pg_config.cam_mask import CamMask
from pgdrive.world.image_buffer import ImageBuffer
from pgdrive.world.pg_world import PGWorld


class RGBCamera(ImageBuffer):
    # shape(dim_1, dim_2)
    BUFFER_W = 84  # dim 1
    BUFFER_H = 84  # dim 2
    CAM_MASK = CamMask.RgbCam
    default_region = [1 / 3, 2 / 3, ImageBuffer.display_bottom, ImageBuffer.display_top]

    def __init__(self, length: int, width: int, chassis_np: NodePath, pg_world: PGWorld):
        self.BUFFER_W = length
        self.BUFFER_H = width
        super(RGBCamera, self).__init__(
            self.BUFFER_W,
            self.BUFFER_H,
            Vec3(0.0, 0.8, 1.5),
            self.BKG_COLOR,
            pg_world=pg_world,
            parent_node=chassis_np
        )
        self.add_to_display(pg_world, self.default_region)
        self.cam.lookAt(0, 2.4, 1.3)
        self.lens.setFov(60)
        self.lens.setAspectRatio(2.0)
