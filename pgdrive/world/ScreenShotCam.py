from panda3d.core import Vec3
from pgdrive.pg_config.cam_mask import CamMask
from pgdrive.scene_creator.ego_vehicle.vehicle_module.mini_map import MiniMap


class ScreenShotCam(MiniMap):
    CAM_MASK = CamMask.ScreenshotCam
    BKG_COLOR = (0, 0, 0, 0)

    def __init__(self, buffer_w, buffer_h, position, height, parent_np, pg_world):
        super(ScreenShotCam, self).__init__((buffer_w, buffer_h, height), parent_np, pg_world)
        self.cam.setPos(Vec3(*position, height))
        self.cam.lookAt(Vec3(*position, 0))
        self.lens.setAspectRatio(1.0)
        self.lens.setFov(90)

    def add_to_display(self, pg_world, display_region):
        """
        Give up automatic adding to display
        """
        return
