from panda3d.core import Vec3, NodePath

from pgdrive.pg_config.cam_mask import CamMask
from pgdrive.world.image_buffer import ImageBuffer
from pgdrive.world.pg_world import PgWorld


class RgbCamera(ImageBuffer):
    # shape(dim_1, dim_2)
    BUFFER_X = 84  # dim 1
    BUFFER_Y = 84  # dim 2
    CAM_MASK = CamMask.RgbCam
    display_top = 1.0

    def __init__(self, length: int, width: int, chassis_np: NodePath, pg_world: PgWorld):
        self.BUFFER_X = length
        self.BUFFER_Y = width
        super(RgbCamera, self).__init__(
            self.BUFFER_X, self.BUFFER_Y, Vec3(0.0, 0.8, 1.5), self.BKG_COLOR, pg_world.win, pg_world.makeCamera,
            chassis_np
        )
        self.add_to_display(pg_world, [1 / 3, 2 / 3, self.display_bottom, self.display_top])
        self.cam.lookAt(0, 2.4, 1.3)
        self.lens.setFov(60)
        self.lens.setAspectRatio(2.0)
