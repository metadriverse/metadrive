from panda3d.core import Vec3, NodePath

from pg_drive.pg_config.cam_mask import CamMask
from pg_drive.world.pg_world import PgWorld
from pg_drive.world.image_buffer import ImageBuffer


class SensorCamera(ImageBuffer):
    # shape(dim_1, dim_2)
    BUFFER_X = 84  # dim 1
    BUFFER_Y = 84  # dim 2
    CAM_MASK = CamMask.FrontCam
    display_top = 1.0

    def __init__(self, length: int, width: int, chassis_np: NodePath, pg_world: PgWorld):
        self.BUFFER_X = length
        self.BUFFER_Y = width
        super(SensorCamera, self).__init__(
            self.BUFFER_X, self.BUFFER_Y, Vec3(0.0, 0.8, 1.5), self.BKG_COLOR, pg_world.win.makeTextureBuffer,
            pg_world.makeCamera, chassis_np
        )
        self.add_to_display(pg_world, [0.33, 0.67, self.display_bottom, self.display_top])
        self.cam.lookAt(0, 2.4, 1.3)
        lens = self.cam.node().getLens()
        lens.setFov(60)
        # lens.setAspectRatio(2.0)
