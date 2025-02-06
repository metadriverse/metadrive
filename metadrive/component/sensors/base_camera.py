from typing import Union

import cv2
import numpy as np
from panda3d.core import NodePath

from metadrive.component.sensors.base_sensor import BaseSensor
from metadrive.utils.cuda import check_cudart_err

_cuda_enable = True
try:
    import cupy as cp
    from OpenGL.GL import GL_TEXTURE_2D  # noqa F403
    from cuda import cudart
    from cuda.cudart import cudaGraphicsRegisterFlags
    from panda3d.core import GraphicsOutput, Texture, GraphicsStateGuardianBase, DisplayRegionDrawCallbackData
except ImportError:
    _cuda_enable = False
from panda3d.core import Vec3

from metadrive.engine.core.image_buffer import ImageBuffer
from metadrive import constants


class BaseCamera(ImageBuffer, BaseSensor):
    """
    This class wrapping the ImageBuffer and BaseSensor to implement perceive() function to capture images in the virtual
    world. It also extends a support for cuda, so the rendered images can be retained on GPU and converted to torch
    tensor directly. The sensor is shared and thus can be set at any position in the world for any objects' use.
    To enable the image observation, set image_observation to True.
    """
    # shape(dim_1, dim_2)
    BUFFER_W = 84  # dim 1
    BUFFER_H = 84  # dim 2
    CAM_MASK = None

    num_channels = 3

    def __init__(self, engine, need_cuda=False, frame_buffer_property=None):
        self._enable_cuda = need_cuda
        super(BaseCamera, self).__init__(
            self.BUFFER_W, self.BUFFER_H, self.BKG_COLOR, engine=engine, frame_buffer_property=frame_buffer_property
        )

        width = self.BUFFER_W
        height = self.BUFFER_H
        if (width > 100 or height > 100) and not self.enable_cuda:
            # Too large height or width will cause corruption in Mac.
            self.logger.warning(
                "You are using too large buffer! The height is {}, and width is {}. "
                "It may lower the sample efficiency! Consider reducing buffer size or use cuda image by"
                " set [image_on_cuda=True].".format(height, width)
            )
        self.cuda_graphics_resource = None
        if self.enable_cuda:
            assert _cuda_enable, "Can not enable cuda rendering pipeline, if you are on Windows, try 'pip install pypiwin32'"

            # returned tensor property
            self.cuda_dtype = np.uint8
            self.cuda_shape = (self.BUFFER_W, self.BUFFER_H)
            self.cuda_strides = None
            self.cuda_order = "C"

            self._cuda_buffer = None

            # make texture
            self._make_cuda_texture()

            def _callback_func(cbdata: DisplayRegionDrawCallbackData):
                # print("DRAW CALLBACK!!!!!!!!!!!!!!!11")
                cbdata.upcall()
                if not self.registered and self.texture_context_future.done():
                    self.register()
                if self.registered:
                    with self as array:
                        self.cuda_rendered_result = array

            # Fill the buffer due to multi-thread
            for _ in range(3):
                self.engine.graphicsEngine.renderFrame()
            self.cam.node().getDisplayRegion(0).setDrawCallback(_callback_func)

            self.gsg = GraphicsStateGuardianBase.getDefaultGsg()
            self.texture_context_future = self.cuda_texture.prepare(self.gsg.prepared_objects)
            self.cuda_texture_identifier = None
            self.new_cuda_mem_ptr = None
            self.cuda_rendered_result = None

    def _make_cuda_texture(self):
        """
        Make a texture for cuda access
        """
        self.cuda_texture = Texture()
        self.buffer.addRenderTexture(self.cuda_texture, GraphicsOutput.RTMBindOrCopy)

    @property
    def enable_cuda(self):
        return self is not None and self._enable_cuda

    def get_image(self, base_object, mode="bgr"):
        """
        Put camera to an object and get the image.
        """
        original_parent = self.cam.getParent()
        original_position = self.cam.getPos()
        original_hpr = self.cam.getHpr()
        self.cam.reparentTo(base_object.origin)
        img = self.get_rgb_array_cpu()
        self.track(original_parent, original_position, original_hpr)
        if mode == "bgr":
            return img
        elif mode == "rgb":
            return img[..., ::-1]
        else:
            raise ValueError("Unknown mode: {}".format(mode))

    def save_image(self, base_object, name="debug.png"):
        """
        Put camera to an object and save the image to the disk
        """
        img = self.get_image(base_object, mode="bgr")
        cv2.imwrite(name, img)

    def track(self, new_parent_node: NodePath, position, hpr):
        """
        Track a given object.
        """
        self.cam.reparentTo(new_parent_node)
        self.cam.setPos(*position)
        self.cam.setHpr(*hpr)

    def perceive(
        self, to_float=True, new_parent_node: Union[NodePath, None] = None, position=None, hpr=None
    ) -> np.ndarray:
        """
        When to_float is set to False, the image will be represented by unit8 with component value ranging from [0-255].
        Otherwise, it will be float type with component value ranging from [0.-1.]. By default, the reset parameters are
        all None. In this case, the camera will render the result with poses and position set by track() function.

        When the reset parameters are not None, this camera will be mounted to a new place and render corresponding new
        results. After this, the camera will be returned to the original states. This process is like borrowing the
        camera to capture a new image and return the camera to the owner. This usually happens when using one camera to
        render multiple times from different positions and poses.

        new_parent_node should be a NodePath like object.origin or vehicle.origin or self.engine.origin, which
        means the world origin. When new_parent_node is set, both position and hpr have to be set as well. The position
        and hpr are all 3-dim vector representing:
            1) the relative position to the reparent node
            2) the heading/pitch/roll of the sensor

        Args:
            to_float: When to_float is set to False, the image will be represented by unit8 with component value ranging
                from [0-255]. Otherwise, it will be float type with component value ranging from [0.-1.].
            new_parent_node: new_parent_node should be a NodePath like object.origin or vehicle.origin or
                self.engine.origin, which means the world origin. When new_parent_node is set, both position and hpr
                have to be set as well. The position and hpr are all 3-dim vector representing:
            position: the relative position to the reparent node
            hpr: the heading/pitch/roll of the sensor

        Return:
            Array representing the image.
        """

        different_pos_hpr = False
        if new_parent_node:
            if position is None:
                position = constants.DEFAULT_SENSOR_OFFSET
            if hpr is None:
                hpr = constants.DEFAULT_SENSOR_HPR

            # return camera to original state
            original_object = self.cam.getParent()
            original_hpr = self.cam.getHpr()
            original_position = self.cam.getPos()

            # reparent to new parent node
            self.cam.reparentTo(new_parent_node)
            # relative position
            assert len(position) == 3, "The first parameter of camera.perceive() should be a BaseObject instance " \
                                       "or a 3-dim vector representing the (x,y,z) position."
            self.cam.setPos(Vec3(*position))
            assert len(hpr) == 3, "The hpr parameter of camera.perceive() should be  a 3-dim vector representing " \
                                  "the heading/pitch/roll."
            self.cam.setHpr(Vec3(*hpr))

            different_pos_hpr = (original_hpr != Vec3(*hpr)) or (original_position != Vec3(*position))

            self.engine.taskMgr.step()

        if different_pos_hpr:
            # Step the engine to call a new "self.engine.graphicsEngine.renderFrame()"
            # (not sure why need to step twice...
            self.engine.taskMgr.step()

        if self.enable_cuda:
            assert self.cuda_rendered_result is not None
            ret = self.cuda_rendered_result[..., :self.num_channels][..., ::-1][::-1]
        else:
            ret = self.get_rgb_array_cpu()

        if new_parent_node:
            # return camera to original objects
            self.cam.reparentTo(original_object)
            self.cam.setHpr(original_hpr)
            self.cam.setPos(original_position)
        return self._format(ret, to_float)

    def _format(self, ret, to_float):
        """
        Format the image to the desired type, float32 or uint8
        """
        if not to_float:
            return ret.astype(np.uint8, copy=False, order="C")
        else:
            return ret / 255

    def destroy(self):
        if self.registered:
            self.unregister()
        ImageBuffer.destroy(self)

    def get_cam(self):
        return self.cam

    def get_lens(self):
        return self.lens

    # # following functions are for onscreen render
    # def add_display_region(self, display_region):
    #     super(BaseCamera, self).add_display_region(display_region)

    def remove_display_region(self):
        super(BaseCamera, self).remove_display_region()

    def __del__(self):
        if self.enable_cuda:
            self.unregister()
        super(BaseCamera, self).__del__()

    """
    Following functions are cuda support
    """

    @property
    def cuda_array(self):
        assert self.mapped
        return cp.ndarray(
            shape=(self.cuda_shape[1], self.cuda_shape[0], 4),
            dtype=self.cuda_dtype,
            strides=self.cuda_strides,
            order=self.cuda_order,
            memptr=self._cuda_buffer
        )

    @property
    def cuda_buffer(self):
        assert self.mapped
        return self._cuda_buffer

    @property
    def graphics_resource(self):
        assert self.registered
        return self.cuda_graphics_resource

    @property
    def registered(self):
        return self.cuda_graphics_resource is not None

    @property
    def mapped(self):
        return self._cuda_buffer is not None

    def __enter__(self):
        return self.map()

    def __exit__(self, exc_type, exc_value, trace):
        self.unmap()
        return False

    def register(self):
        self.cuda_texture_identifier = self.texture_context_future.result().getNativeId()
        assert self.cuda_texture_identifier is not None
        if self.registered:
            return self.cuda_graphics_resource
        self.cuda_graphics_resource = check_cudart_err(
            cudart.cudaGraphicsGLRegisterImage(
                self.cuda_texture_identifier, GL_TEXTURE_2D, cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly
            )
        )
        return self.cuda_graphics_resource

    def unregister(self):
        if self.registered:
            self.unmap()
            self.cuda_graphics_resource = check_cudart_err(
                cudart.cudaGraphicsUnregisterResource(self.cuda_graphics_resource)
            )
            self.cam.node().getDisplayRegion(0).clearDrawCallback()

    def map(self, stream=0):
        if not self.registered:
            raise RuntimeError("Cannot map an unregistered buffer.")
        if self.mapped:
            return self._cuda_buffer
        check_cudart_err(cudart.cudaGraphicsMapResources(1, self.cuda_graphics_resource, stream))
        array = check_cudart_err(cudart.cudaGraphicsSubResourceGetMappedArray(self.graphics_resource, 0, 0))
        channelformat, cudaextent, flag = check_cudart_err(cudart.cudaArrayGetInfo(array))

        depth = 1
        byte = 4  # four channel
        if self.new_cuda_mem_ptr is None:
            success, self.new_cuda_mem_ptr = cudart.cudaMalloc(cudaextent.height * cudaextent.width * byte * depth)
        check_cudart_err(
            cudart.cudaMemcpy2DFromArray(
                self.new_cuda_mem_ptr, cudaextent.width * byte * depth, array, 0, 0, cudaextent.width * byte * depth,
                cudaextent.height, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
            )
        )
        if self._cuda_buffer is None:
            self._cuda_buffer = cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(self.new_cuda_mem_ptr, cudaextent.width * depth * byte * cudaextent.height, self),
                0
            )
        return self.cuda_array

    def unmap(self, stream=None):
        if not self.registered:
            raise RuntimeError("Cannot unmap an unregistered buffer.")
        if not self.mapped:
            return self
        self._cuda_buffer = check_cudart_err(cudart.cudaGraphicsUnmapResources(1, self.cuda_graphics_resource, stream))
        return self
