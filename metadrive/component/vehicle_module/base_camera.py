import numpy as np
from panda3d.core import Vec3

from metadrive.engine.core.image_buffer import ImageBuffer


class BaseCamera(ImageBuffer):
    """
    To enable the image observation, set offscreen render to True. The instance of subclasses will be singleton, so that
    every objects share the same camera to boost the efficiency and save memory. Camera configuration is read from the
    global config automatically.
    """
    # shape(dim_1, dim_2)
    BUFFER_W = 84  # dim 1
    BUFFER_H = 84  # dim 2
    CAM_MASK = None
    display_region_size = [1 / 3, 2 / 3, ImageBuffer.display_bottom, ImageBuffer.display_top]
    _singleton = None

    attached_object = None

    @classmethod
    def initialized(cls):
        return True if cls._singleton is not None else False

    def __init__(self):
        if not self.initialized():
            super(BaseCamera, self).__init__(self.BUFFER_W, self.BUFFER_H, Vec3(0.0, 0.8, 1.5), self.BKG_COLOR)
            type(self)._singleton = self
            self.init_num = 0
        else:
            type(self)._singleton.init_num += 1

    def get_image(self, base_object):
        """
        Borrow the camera to get observations
        """
        type(self)._singleton.origin.reparentTo(base_object.origin)
        ret = super(BaseCamera, type(self)._singleton).get_image()
        self.track(self.attached_object)
        return ret

    def save_image(self, base_object, name="debug.png"):
        img = self.get_image(base_object)
        img.write(name)

    def get_pixels_array(self, base_object, clip=True) -> np.ndarray:
        img = self.get_image(base_object)
        return type(self)._singleton.convert_to_array(img, clip)

    def destroy(self):
        if self.initialized():
            if type(self)._singleton.init_num > 1:
                type(self)._singleton.init_num -= 1
            else:
                assert type(self)._singleton.init_num == 1
                ImageBuffer.destroy(type(self)._singleton)
                type(self)._singleton = None

    def get_cam(self):
        return type(self)._singleton.cam

    def get_lens(self):
        return type(self)._singleton.lens

    # following functions are for onscreen render
    def add_display_region(self, display_region):
        self.track(self.attached_object)
        super(BaseCamera, self).add_display_region(display_region)

    def remove_display_region(self):
        super(BaseCamera, self).remove_display_region()

    def track(self, base_object):
        if base_object is not None:
            self.attached_object = base_object
            type(self)._singleton.origin.reparentTo(base_object.origin)
