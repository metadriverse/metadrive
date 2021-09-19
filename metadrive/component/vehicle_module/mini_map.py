from panda3d.core import Vec3

from metadrive.component.vehicle_module.base_camera import BaseCamera
from metadrive.constants import CamMask
from metadrive.engine.engine_utils import get_global_config, engine_initialized


class MiniMap(BaseCamera):
    CAM_MASK = CamMask.MiniMap
    display_region_size = [0., 1 / 3, BaseCamera.display_bottom, BaseCamera.display_top]

    def __init__(self):
        assert engine_initialized(), "You should initialize engine before adding camera to vehicle"
        config = get_global_config()["vehicle_config"]["mini_map"]
        self.BUFFER_W, self.BUFFER_H = config[0], config[1]
        height = config[2]
        super(MiniMap, self).__init__()

        cam = self.get_cam()
        lens = self.get_lens()

        cam.setZ(height)
        cam.lookAt(Vec3(0, 20, 0))
        lens.setAspectRatio(2.0)
