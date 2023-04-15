import numpy as np
import logging

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


class BaseCamera(ImageBuffer):
    """
    To enable the image observation, set image_observation to True. The instance of subclasses will be singleton, so that
    every objects share the same camera, to boost the efficiency and save memory. Camera configuration is read from the
    global config automatically.
    """
    # shape(dim_1, dim_2)
    BUFFER_W = 84  # dim 1
    BUFFER_H = 84  # dim 2
    CAM_MASK = None
    display_region_size = [1 / 3, 2 / 3, 0.8, 1.0]
    _singleton = None

    attached_object = None

    @classmethod
    def initialized(cls):
        return True if cls._singleton is not None else False

    def __init__(self, setup_pbr=False, need_cuda=False):
        if not self.initialized():
            super(BaseCamera, self).__init__(
                self.BUFFER_W, self.BUFFER_H, Vec3(0.8, 0., 1.5), self.BKG_COLOR, setup_pbr=setup_pbr
            )
            type(self)._singleton = self
            self.init_num = 1
            self._enable_cuda = self.engine.global_config["image_on_cuda"] and need_cuda

            width = self.BUFFER_W
            height = self.BUFFER_H
            if (width > 100 or height > 100) and not self.enable_cuda:
                # Too large height or width will cause corruption in Mac.
                logging.warning(
                    "You may using too large buffer! The height is {}, and width is {}. "
                    "It may lower the sample efficiency! Considering reduce buffer size or using cuda image by"
                    " set [image_on_cuda=True].".format(height, width)
                )
            self.cuda_graphics_resource = None
            if self.enable_cuda:
                assert _cuda_enable, "Can not enable cuda rendering pipeline"

                # returned tensor property
                self.cuda_dtype = np.uint8
                self.cuda_shape = (self.BUFFER_W, self.BUFFER_H)
                self.cuda_strides = None
                self.cuda_order = "C"

                self._cuda_buffer = None

                # make texture
                self.cuda_texture = Texture()
                self.buffer.addRenderTexture(self.cuda_texture, GraphicsOutput.RTMBindOrCopy)

                def _callback_func(cbdata: DisplayRegionDrawCallbackData):
                    # print("DRAW CALLBACK!!!!!!!!!!!!!!!11")
                    cbdata.upcall()
                    if not type(self)._singleton.registered and type(self)._singleton.texture_context_future.done():
                        type(self)._singleton.register()
                    if type(self)._singleton.registered:
                        with type(self)._singleton as array:
                            type(self)._singleton.cuda_rendered_result = array

                # Fill the buffer due to multi-thread
                self.engine.graphicsEngine.renderFrame()
                self.engine.graphicsEngine.renderFrame()
                self.engine.graphicsEngine.renderFrame()
                self.cam.node().getDisplayRegion(0).setDrawCallback(_callback_func)

                self.gsg = GraphicsStateGuardianBase.getDefaultGsg()
                self.texture_context_future = self.cuda_texture.prepare(self.gsg.prepared_objects)
                self.cuda_texture_identifier = None
                self.new_cuda_mem_ptr = None
                self.cuda_rendered_result = None

        else:
            type(self)._singleton.init_num += 1

    @property
    def enable_cuda(self):
        return type(self)._singleton is not None and type(self)._singleton._enable_cuda

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
        self.track(base_object)
        if self.enable_cuda:
            assert type(self)._singleton.cuda_rendered_result is not None
            ret = type(self)._singleton.cuda_rendered_result[..., :-1][..., ::-1][::-1]
        else:
            ret = type(self)._singleton.get_rgb_array()
        if self.engine.global_config["vehicle_config"]["rgb_to_grayscale"]:
            ret = np.dot(ret[..., :3], [0.299, 0.587, 0.114])
        if not clip:
            return ret.astype(np.uint8)
        else:
            return ret / 255

    def destroy(self):
        if self.initialized():
            if type(self)._singleton.init_num > 1:
                type(self)._singleton.init_num -= 1
            elif type(self)._singleton.init_num == 0:
                raise RuntimeError("No {}, can not destroy".format(self.__class__.__name__))
            else:
                type(self).init_num = 0
                assert type(self)._singleton.init_num == 1 or type(self)._singleton.init_num == 0
                if type(self)._singleton is not None and type(self)._singleton.registered:
                    self.unregister()
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
        if base_object is not None and type(self)._singleton is not None:
            self.attached_object = base_object
            type(self)._singleton.origin.reparentTo(base_object.origin)

    def __del__(self):
        if self.enable_cuda:
            type(self)._singleton.unregister()
        type(self)._singleton = None
        super(BaseCamera, self).__del__()

    """
    Following functions are cuda support
    """

    @property
    def cuda_array(self):
        assert type(self)._singleton.mapped
        return cp.ndarray(
            shape=(type(self)._singleton.cuda_shape[1], type(self)._singleton.cuda_shape[0], 4),
            dtype=type(self)._singleton.cuda_dtype,
            strides=type(self)._singleton.cuda_strides,
            order=type(self)._singleton.cuda_order,
            memptr=type(self)._singleton._cuda_buffer
        )

    @property
    def cuda_buffer(self):
        assert type(self)._singleton.mapped
        return type(self)._singleton._cuda_buffer

    @property
    def graphics_resource(self):
        assert type(self)._singleton.registered
        return type(self)._singleton.cuda_graphics_resource

    @property
    def registered(self):
        return type(self)._singleton.cuda_graphics_resource is not None

    @property
    def mapped(self):
        return type(self)._singleton._cuda_buffer is not None

    def __enter__(self):
        return type(self)._singleton.map()

    def __exit__(self, exc_type, exc_value, trace):
        type(self)._singleton.unmap()
        return False

    def register(self):
        type(self)._singleton.cuda_texture_identifier = type(self
                                                             )._singleton.texture_context_future.result().getNativeId()
        assert type(self)._singleton.cuda_texture_identifier is not None
        if type(self)._singleton.registered:
            return type(self)._singleton.cuda_graphics_resource
        type(self)._singleton.cuda_graphics_resource = check_cudart_err(
            cudart.cudaGraphicsGLRegisterImage(
                type(self)._singleton.cuda_texture_identifier, GL_TEXTURE_2D,
                cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly
            )
        )
        return type(self)._singleton.cuda_graphics_resource

    def unregister(self):
        if type(self)._singleton.registered:
            type(self)._singleton.unmap()
            type(self)._singleton.cuda_graphics_resource = check_cudart_err(
                cudart.cudaGraphicsUnregisterResource(type(self)._singleton.cuda_graphics_resource)
            )
            self.cam.node().getDisplayRegion(0).clearDrawCallback()

    def map(self, stream=0):
        if not type(self)._singleton.registered:
            raise RuntimeError("Cannot map an unregistered buffer.")
        if type(self)._singleton.mapped:
            return type(self)._singleton._cuda_buffer
        # self.engine.graphicsEngine.renderFrame()
        check_cudart_err(cudart.cudaGraphicsMapResources(1, type(self)._singleton.cuda_graphics_resource, stream))
        array = check_cudart_err(
            cudart.cudaGraphicsSubResourceGetMappedArray(type(self)._singleton.graphics_resource, 0, 0)
        )
        channelformat, cudaextent, flag = check_cudart_err(cudart.cudaArrayGetInfo(array))

        depth = 1
        byte = 4  # four channel
        if type(self)._singleton.new_cuda_mem_ptr is None:
            success, type(self)._singleton.new_cuda_mem_ptr = cudart.cudaMalloc(
                cudaextent.height * cudaextent.width * byte * depth
            )
        check_cudart_err(
            cudart.cudaMemcpy2DFromArray(
                type(self)._singleton.new_cuda_mem_ptr, cudaextent.width * byte * depth, array, 0, 0,
                cudaextent.width * byte * depth, cudaextent.height, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
            )
        )
        if type(self)._singleton._cuda_buffer is None:
            type(self)._singleton._cuda_buffer = cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(
                    type(self)._singleton.new_cuda_mem_ptr, cudaextent.width * depth * byte * cudaextent.height,
                    type(self)._singleton
                ), 0
            )
        return type(self)._singleton.cuda_array

    def unmap(self, stream=None):
        if not type(self)._singleton.registered:
            raise RuntimeError("Cannot unmap an unregistered buffer.")
        if not type(self)._singleton.mapped:
            return type(self)._singleton
        type(self)._singleton._cuda_buffer = check_cudart_err(
            cudart.cudaGraphicsUnmapResources(1,
                                              type(self)._singleton.cuda_graphics_resource, stream)
        )
        return type(self)._singleton
