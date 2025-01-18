from typing import Union, List

import numpy as np
from panda3d.core import NodePath, Vec3, Vec4, Camera

from metadrive.constants import RENDER_MODE_ONSCREEN, BKG_COLOR, RENDER_MODE_NONE
from metadrive.engine.logger import get_logger


class ImageBuffer:
    """
    This is a wrapper for FrameBuffer, associated with a camera. The camera scene in the camera view will be rendered
    into the buffer. Thus, we can access the image in the buffer and can apply effect to the image to implement
    DepthCamera, SemanticCamera and So on. It also allows opening a display region on the main window to show sensor
    output.
    """
    LINE_FRAME_COLOR = (0.8, 0.8, 0.8, 0)
    CAM_MASK = None
    BUFFER_W = 84  # left to right
    BUFFER_H = 84  # bottom to top
    BKG_COLOR = BKG_COLOR
    # display_bottom = 0.8
    # display_top = 1
    display_region = None
    line_borders = []
    num_channels = 3

    def __init__(
        self,
        width: float,
        height: float,
        bkg_color: Union[Vec4, Vec3],
        parent_node: NodePath = None,
        frame_buffer_property=None,
        engine=None
    ):
        self.logger = get_logger()
        self._node_path_list = []

        # from metadrive.engine.engine_utils import get_engine
        self.engine = engine
        try:
            assert self.engine.win is not None, "{} cannot be made without use_render or image_observation".format(
                self.__class__.__name__
            )
            assert self.CAM_MASK is not None, "Define a camera mask for every image buffer"
        except AssertionError:
            self.logger.debug("Cannot create {}".format(self.__class__.__name__))
            self.buffer = None
            self.cam = NodePath(Camera("non-sense camera"))
            self._node_path_list.append(self.cam)

            self.lens = self.cam.node().getLens()
            return

        # self.texture = Texture()
        self._create_buffer(width, height, frame_buffer_property)
        self._create_camera(parent_node, bkg_color)

        self._setup_effect()
        self.logger.debug("Load Image Buffer: {}".format(self.__class__.__name__))

    def _create_camera(self, parent_node, bkg_color):
        """
        Create camera for the buffer
        """
        self.cam = cam = self.engine.makeCamera(self.buffer, clearColor=bkg_color)
        cam.node().setCameraMask(self.CAM_MASK)
        if parent_node:
            self.cam.reparentTo(parent_node)
        self.lens = self.cam.node().getLens()
        self.lens.setFov(self.engine.global_config["camera_fov"])
        self.lens.setAspectRatio(self.BUFFER_W / self.BUFFER_H)

    def _create_buffer(self, width, height, frame_buffer_property):
        """
        The buffer is created without frame_buffer_property by default
        Args:
            width: Image width
            height: Image height
            frame_buffer_property: disabled in Semantic Camera

        Returns: Buffer object

        """
        if frame_buffer_property is not None:
            self.buffer = self.engine.win.makeTextureBuffer(
                self.__class__.__name__, width, height, fbp=frame_buffer_property
            )
        else:
            self.buffer = self.engine.win.makeTextureBuffer(self.__class__.__name__, width, height)

    def _setup_effect(self):
        """
        Apply effect to the render the scene. Usually setup shader here
        Returns: None

        """
        pass

    def get_rgb_array_cpu(self):
        """
        Get the rgb array on CPU, which suffers from the latency of moving data from graphics card to memory
        """
        origin_img = self.buffer.getDisplayRegion(1).getScreenshot()
        img = np.frombuffer(origin_img.getRamImage().getData(), dtype=np.uint8)
        img = img.reshape((origin_img.getYSize(), origin_img.getXSize(), -1))
        img = img[..., :self.num_channels]
        img = img[::-1]
        return img

    @staticmethod
    def get_grayscale_array(img, clip=True):
        raise DeprecationWarning("This API is deprecated")
        if not clip:
            numpy_array = np.array(
                [[int(img.getGray(i, j) * 255) for j in range(img.getYSize())] for i in range(img.getXSize())],
                dtype=np.uint8
            )
            return np.clip(numpy_array, 0, 255)
        else:
            numpy_array = np.array([[img.getGray(i, j) for j in range(img.getYSize())] for i in range(img.getXSize())])
            return np.clip(numpy_array, 0, 1)

    def add_display_region(self, display_region: List[float], keep_height=True):
        """
        Make a display region for this image buffer and show in on main window
        """
        # only show them when onscreen
        if self.engine.mode != RENDER_MODE_NONE and self.display_region is None:
            if keep_height:
                ratio = self.BUFFER_H / self.BUFFER_W
                h = 0.333 * ratio
                display_region[-2] = 1 - h

            self.display_region = self.engine.win.makeDisplayRegion(*display_region)
            self.display_region.setCamera(self.buffer.getDisplayRegions()[1].camera)
            self.draw_border(display_region)

    def draw_border(self, display_region):
        engine = self.engine
        # add white frame for display region, convert to [-1, 1]
        left = display_region[0] * 2 - 1
        right = display_region[1] * 2 - 1
        bottom = display_region[2] * 2 - 1
        top = display_region[3] * 2 - 1

        self.line_borders.append(engine.draw_line_2d([left, bottom], [left, top], self.LINE_FRAME_COLOR, 1.5))
        self.line_borders.append(engine.draw_line_2d([left, top], [right, top], self.LINE_FRAME_COLOR, 1.5))
        self.line_borders.append(engine.draw_line_2d([right, top], [right, bottom], self.LINE_FRAME_COLOR, 1.5))
        self.line_borders.append(engine.draw_line_2d([right, bottom], [left, bottom], self.LINE_FRAME_COLOR, 1.5))

    def remove_display_region(self):
        engine = self.engine
        if engine.mode == RENDER_MODE_ONSCREEN and self.display_region is not None:
            engine.win.removeDisplayRegion(self.display_region)
            self.display_region = None
        for line_node in self.line_borders:
            line_node.detachNode()

    def destroy(self):
        engine = self.engine
        if engine is not None:
            self.remove_display_region()
            if self.buffer is not None:
                engine.graphicsEngine.removeWindow(self.buffer)
            self.display_region = None
            self.buffer = None
            if self.cam in engine.camList:
                engine.camList.remove(self.cam)
        self.cam.removeNode()
        if len(self.line_borders) != 0:
            for line_np in self.line_borders:
                if line_np:
                    line_np.removeNode()
        self.line_borders = []
        if hasattr(self, "origin"):
            self.origin.removeNode()

        from metadrive.base_class.base_object import clear_node_list
        clear_node_list(self._node_path_list)

    def __del__(self):
        self.logger.debug("{} is destroyed".format(self.__class__.__name__))
