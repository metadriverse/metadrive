import logging
from typing import Union, List

import numpy as np
from panda3d.core import NodePath, Vec3, Vec4, Camera


class ImageBuffer:
    enable = True

    CAM_MASK = None
    BUFFER_X = 800  # left to right
    BUFFER_Y = 800  # bottom to top
    BKG_COLOR = Vec3(179 / 255, 211 / 255, 216 / 255)
    display_bottom = 0.8
    display_top = 1
    display_region = None
    refresh_frame = None

    def __init__(
        self,
        length: float,
        width: float,
        pos: Vec3,
        bkg_color: Union[Vec4, Vec3],
        pg_world_win,
        make_camera_func,
        parent_node: NodePath,
        frame_buffer_property=None
    ):
        try:
            assert ImageBuffer.enable, "Image buffer cannot be created, since the panda3d render pipeline is not loaded"
            assert pg_world_win is not None, "{} cannot be made without use_render or use_image".format(
                self.__class__.__name__
            )
            assert self.CAM_MASK is not None, "Define a camera mask for every image buffer"
        except AssertionError:
            logging.debug("Cannot create {}, maybe the render pipe is highway render".format(self.__class__.__name__))
            self.buffer = None
            self.cam = NodePath(Camera("non-sense camera"))
            self.lens = self.cam.node().getLens()
            return
        make_buffer_func = pg_world_win.makeTextureBuffer
        # self.texture = Texture()
        if frame_buffer_property is None:
            self.buffer = make_buffer_func("camera", length, width)
        else:
            self.buffer = make_buffer_func("camera", length, width, fbp=frame_buffer_property)
            # now we have to setup a new scene graph to make this scene

        self.node_path = NodePath("new render")

        # this takes care of setting up their camera properly
        self.cam = make_camera_func(self.buffer, clearColor=bkg_color)
        self.cam.reparentTo(self.node_path)
        self.cam.setPos(pos)
        self.lens = self.cam.node().getLens()
        self.cam.node().setCameraMask(self.CAM_MASK)
        self.node_path.reparentTo(parent_node)
        logging.debug("Load Image Buffer: {}".format(self.__class__.__name__))

    def get_image(self):
        """
        Bugs here! when use offscreen mode, thus the front cam obs is not from front cam now
        """
        self.refresh_frame()
        from panda3d.core import PNMImage
        img = PNMImage()
        self.buffer.getScreenshot(img)
        return img

    def save_image(self):
        """
        for debug use
        """
        img = self.get_image()
        img.write("debug.jpg")

    def get_pixels_array(self, clip=True) -> np.ndarray:
        """
        default: For gray scale image, one channel. Override this func, when you want a new obs type
        """
        img = self.get_image()

        if not clip:
            numpy_array = np.array(
                [[int(img.getGray(i, j) * 255) for j in range(img.getYSize())] for i in range(img.getXSize())],
                dtype=np.uint8
            )
            return np.clip(numpy_array, 0, 255)
        else:
            numpy_array = np.array([[img.getGray(i, j) for j in range(img.getYSize())] for i in range(img.getXSize())])
            return np.clip(numpy_array, 0, 1)

    def add_to_display(self, pg_world, display_region: List[float]):
        if not self.enable:
            return
        if pg_world.pg_config["use_render"]:
            # only show them when onscreen
            self.display_region = pg_world.win.makeDisplayRegion(*display_region)
            self.display_region.setCamera(self.cam)
            pg_world.my_display_regions.append(self.display_region)
            pg_world.my_buffers.append(self)

    def __del__(self):
        logging.debug("{} is destroyed".format(self.__class__.__name__))
