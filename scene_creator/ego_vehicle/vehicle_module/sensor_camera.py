from world.bt_world import BtWorld
from world.ImageBuffer import ImageBuffer
from panda3d.core import Vec3
from pg_config.cam_mask import CamMask


class SensorCamera(ImageBuffer):
    BUFFER_L = 84
    BUFFER_W = 84
    CAM_MASK = CamMask.FrontCam
    display_top = 1.0

    def __init__(self, chassis_np: float, bt_world: BtWorld):
        super(SensorCamera, self).__init__(
            self.BUFFER_L, self.BUFFER_W, Vec3(0.0, 0.8, 0.73), self.BKG_COLOR, bt_world.win.makeTextureBuffer,
            bt_world.makeCamera, bt_world.render
        )
        bt_world.add_to_console(self, [0.33, 0.67, self.display_bottom, self.display_top])
        self.cam.reparentTo(chassis_np)
        self.cam.setPos(0.0, 0.8, 1.5)
        self.cam.lookAt(0, 2.4, 1.2)
        lens = self.cam.node().getLens()
        lens.setFov(60)
        lens.setAspectRatio(2.0)
        lens.setFar(300)
