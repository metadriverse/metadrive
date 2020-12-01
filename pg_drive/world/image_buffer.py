import logging
from typing import Union, List

import numpy as np
from panda3d.core import NodePath, Vec3, Vec4


class ImageBuffer:
    CAM_MASK = None
    BUFFER_X = 800  # left to right
    BUFFER_Y = 800  # bottom to top
    BKG_COLOR = Vec3(179 / 255, 211 / 255, 216 / 255)
    display_bottom = 0.8
    display_top = 1

    def __init__(
        self, length: float, width: float, pos: Vec3, bkg_color: Union[Vec4, Vec3], make_buffer_func, make_camera_func,
        parent_node: NodePath
    ):
        assert self.CAM_MASK is not None, "define a camera mask for every image buffer"
        # self.texture = Texture()
        self.buffer = make_buffer_func("camera", length, width)
        # now we have to setup a new scene graph to make this scene

        self.node_path = NodePath("new render")

        # this takes care of setting up their camera properly
        self.cam = make_camera_func(self.buffer, clearColor=bkg_color)
        self.cam.reparentTo(self.node_path)
        self.cam.setPos(pos)
        self.cam.node().setCameraMask(self.CAM_MASK)
        self.node_path.reparentTo(parent_node)

    def get_image(self):
        """
        Bugs here! when use offscreen mode, thus the front cam obs is not from front cam now
        """
        from panda3d.core import PNMImage
        img = PNMImage()
        self.buffer.getScreenshot(img)
        return img

    def save_image(self):
        """
        for debug use
        """
        from panda3d.core import PNMImage
        img = PNMImage()
        self.buffer.getScreenshot(img)
        img.write("debug.jpg")

    def get_gray_pixels_array(self, clip) -> np.ndarray:
        img = self.get_image()
        img.makeGrayscale()
        if not clip:
            numpy_array = np.array(
                [[int(img.getGray(i, j) * 255) for j in range(img.getYSize())] for i in range(img.getXSize())],
                dtype=np.uint8
            )
            return np.clip(numpy_array, 0, 255)
        else:
            numpy_array = np.array([[img.getGray(i, j) for j in range(img.getYSize())] for i in range(img.getXSize())])
            return np.clip(numpy_array, 0, 1)

    def add_to_display(self, bt_world, display_region: List[float]):
        if bt_world.bt_config["use_render"]:
            # only show them when onscreen
            region = bt_world.win.makeDisplayRegion(*display_region)
            region.setCamera(self.cam)
            bt_world.my_display_regions.append(region)
            bt_world.my_buffers.append(self)

    def __del__(self):
        logging.debug("{} is destroyed".format(self.__class__.__name__))
