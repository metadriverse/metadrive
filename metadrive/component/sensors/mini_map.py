from panda3d.core import Vec3

from metadrive.component.sensors.base_camera import BaseCamera
from metadrive.constants import CamMask
from panda3d.core import FrameBufferProperties


class MiniMap(BaseCamera):
    CAM_MASK = CamMask.MiniMap
    display_region_size = [0., 1 / 3, 0.8, 1.0]

    def __init__(self, width, height, z_pos, engine, *, cuda=False):
        self.BUFFER_W, self.BUFFER_H, height = width, height, z_pos
        frame_buffer_property = FrameBufferProperties()
        frame_buffer_property.set_rgba_bits(8, 8, 8, 0)  # disable alpha for RGB camera
        super(MiniMap, self).__init__(engine=engine, need_cuda=cuda, frame_buffer_property=frame_buffer_property)

        cam = self.get_cam()
        lens = self.get_lens()

        cam.setZ(height)
        cam.lookAt(Vec3(20, 0, 0))
        lens.setAspectRatio(2.0)
