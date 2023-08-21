from metadrive.component.sensors.base_camera import BaseCamera
from metadrive.constants import CamMask
from metadrive.engine.engine_utils import engine_initialized, get_global_config
from direct.filter.CommonFilters import CommonFilters


class RGBCamera(BaseCamera):
    # shape(dim_1, dim_2)
    BUFFER_W = 84  # dim 1
    BUFFER_H = 84  # dim 2
    CAM_MASK = CamMask.RgbCam
    PBR_ADAPT = False

    def __init__(self, engine, width, height, cuda=False):
        self.BUFFER_W, self.BUFFER_H = width, height
        super(RGBCamera, self).__init__(engine, True, cuda)
        cam = self.get_cam()
        lens = self.get_lens()
        # cam.lookAt(0, 2.4, 1.3)
        cam.lookAt(0, 10.4, 1.6)

        lens.setFov(60)
        # lens.setAspectRatio(2.0)
