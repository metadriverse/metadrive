from panda3d.core import Vec3

from metadrive.component.sensors.base_camera import BaseCamera
from metadrive.constants import CamMask
from panda3d.core import FrameBufferProperties


class MiniMap(BaseCamera):
    CAM_MASK = CamMask.MiniMap

    def __init__(self, width, height, z_pos, engine, *, cuda=False):
        self.BUFFER_W, self.BUFFER_H, height = width, height, z_pos
        super(MiniMap, self).__init__(engine=engine, need_cuda=cuda)

        cam = self.get_cam()
        lens = self.get_lens()

        cam.setZ(height)
        cam.lookAt(Vec3(20, 0, 0))
        lens.setAspectRatio(2.0)
