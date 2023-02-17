import numpy as np

from metadrive.engine.engine_utils import get_engine


class InterfaceCamera:

    @staticmethod
    def get_pixel_array():
        engine = get_engine()
        origin_img = engine.win.getDisplayRegion(0).getScreenshot()
        v = memoryview(origin_img.getRamImage()).tolist()
        img = np.array(v, dtype=np.uint8)
        img = img.reshape((origin_img.getYSize(), origin_img.getXSize(), 4))
        img = img[::-1]
        img = img[..., :-1]
        return img
