from pg_drive.world.bt_world import BtWorld
from pg_drive.world.image_buffer import ImageBuffer
from panda3d.core import Vec3, NodePath
from pg_drive.pg_config.cam_mask import CamMask


class SensorCamera(ImageBuffer):
    # shape(dim_1, dim_2)
    BUFFER_L = 84  # dim 1
    BUFFER_W = 84  # dim 2
    CAM_MASK = CamMask.FrontCam
    display_top = 1.0

    def __init__(self, length: int, width: int, chassis_np: NodePath, bt_world: BtWorld):
        self.BUFFER_L = length
        self.BUFFER_W = width
        super(SensorCamera, self).__init__(
            self.BUFFER_L, self.BUFFER_W, Vec3(0.0, 0.8, 1.5), self.BKG_COLOR, bt_world.win.makeTextureBuffer,
            bt_world.makeCamera, chassis_np
        )
        bt_world.add_to_console(self, [0.33, 0.67, self.display_bottom, self.display_top])
        self.cam.lookAt(0, 2.4, 1.2)
        lens = self.cam.node().getLens()
        lens.setFov(60)
