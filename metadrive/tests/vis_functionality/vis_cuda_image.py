from cuda import cudart

import cupy as cp
from panda3d.core import NodePath, GraphicsOutput, Texture, GraphicsStateGuardianBase
import cupy as cp
import numpy as np
from OpenGL.GL import *  # noqa F403
from cuda import cudart
from cuda.cudart import cudaGraphicsRegisterFlags
from panda3d.core import loadPrcFileData
from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.intersection import InterSection
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.tests.vis_block.vis_block_base import TestBlock, BKG_COLOR
from OpenGL.GL import glGenBuffers


# loadPrcFileData("", "win-size {} {}".format(1024, 1024))

# require:
# 1. pip install cupy-cuda12x
# 2. CUDA-Python
# 3. PyOpenGL
# 4. pyrr
# 5. glfw
#


def format_cudart_err(err):
    return (
        f"{cudart.cudaGetErrorName(err)[1].decode('utf-8')}({int(err)}): "
        f"{cudart.cudaGetErrorString(err)[1].decode('utf-8')}"
    )


def check_cudart_err(args):
    if isinstance(args, tuple):
        assert len(args) >= 1
        err = args[0]
        if len(args) == 1:
            ret = None
        elif len(args) == 2:
            ret = args[1]
        else:
            ret = args[1:]
    else:
        err = args
        ret = None

    assert isinstance(err, cudart.cudaError_t), type(err)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(format_cudart_err(err))

    return ret


class CUDATest:
    def __init__(self, window_type="onscreen"):
        self.engine = engine = TestBlock(window_type=window_type)
        from metadrive.engine.asset_loader import initialize_asset_loader

        initialize_asset_loader(engine)

        global_network = NodeRoadNetwork()
        first = FirstPGBlock(global_network, 3.0, 2, engine.render, engine.world, 20)

        intersection = InterSection(3, first.get_socket(0), global_network, 1)
        print(intersection.construct_block(engine.render, engine.world))

        id = 4
        for socket_idx in range(intersection.SOCKET_NUM):
            block = Curve(id, intersection.get_socket(socket_idx), global_network, id)
            block.construct_block(engine.render, engine.world)
            id += 1

        intersection = InterSection(id, block.get_socket(0), global_network, 1)
        intersection.construct_block(engine.render, engine.world)

        engine.show_bounding_box(global_network)

        # buffer property
        self._dtype = np.uint32
        self._shape = (800, 600, 4)
        self._strides = None
        self._order = "C"

        self._gl_buffer = None
        self._flags = cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsNone

        self._graphics_resource = None
        self._cuda_buffer = None

        # # make buffer
        self.texture = Texture()
        self.texture.setMinfilter(Texture.FTLinear)
        self.texture.setFormat(Texture.FRgba32)
        self.engine.win.addRenderTexture(self.texture, GraphicsOutput.RTMBindOrCopy)

        self.texture_identifier = None
        self.gsg = GraphicsStateGuardianBase.getDefaultGsg()
        self.texture_context_future = self.texture.prepare(self.gsg.prepared_objects)
        self.new_cuda_mem_ptr = None
        #
        # self.origin = NodePath("new render")
        # # this takes care of setting up their camera properly
        # self.cam = self.engine.makeCamera(self.buffer, clearColor=BKG_COLOR)
        # self.cam.reparentTo(self.engine.worldNP)

    @property
    def cuda_array(self):
        assert self.mapped
        return cp.ndarray(
            shape=self._shape,
            dtype=self._dtype,
            strides=self._strides,
            order=self._order,
            memptr=self._cuda_buffer,
        )

    @property
    def gl_buffer(self):
        return self._gl_buffer

    @property
    def cuda_buffer(self):
        assert self.mapped
        return self._cuda_buffer

    @property
    def graphics_resource(self):
        assert self.registered
        return self._graphics_resource

    @property
    def registered(self):
        return self._graphics_resource is not None

    @property
    def mapped(self):
        return self._cuda_buffer is not None

    def __enter__(self):
        return self.map()

    def __exit__(self, exc_type, exc_value, trace):
        self.unmap()
        return False

    def __del__(self):
        self.unregister()

    def register(self):
        assert self.texture_identifier is not None
        if self.registered:
            return self._graphics_resource
        self._graphics_resource = check_cudart_err(cudart.cudaGraphicsGLRegisterImage(self.texture_identifier,
                                                                                      GL_TEXTURE_2D,
                                                                                      cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly))
        return self._graphics_resource

    def unregister(self):
        if not self.registered:
            return self
        self.unmap()
        self._graphics_resource = check_cudart_err(cudart.cudaGraphicsUnregisterResource(self._graphics_resource))
        return self

    def map(self, stream=0):
        if not self.registered:
            raise RuntimeError("Cannot map an unregistered buffer.")
        if self.mapped:
            return self._cuda_buffer

        check_cudart_err(cudart.cudaGraphicsMapResources(1, self._graphics_resource, stream))
        # ptr, size = check_cudart_err(cudart.cudaGraphicsResourceGetMappedPointer(self._graphics_resource))
        array = check_cudart_err(cudart.cudaGraphicsSubResourceGetMappedArray(self.graphics_resource, 0, 0))
        channelformat, cudaextent, flag = check_cudart_err(cudart.cudaArrayGetInfo(array))

        if cudaextent.width == 1024 and cudaextent.height == 1024:
            success, self.new_cuda_mem_ptr = cudart.cudaMalloc(1)
            check_cudart_err(
                cudart.cudaMemcpy2DFromArray(self.new_cuda_mem_ptr, 1, array, 0,
                                             0,
                                             1, 1,
                                             cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice))
            self._cuda_buffer = cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(self.new_cuda_mem_ptr, 1, self), 0)
        else:
            success, self.new_cuda_mem_ptr = cudart.cudaMalloc(
                cudaextent.height * cudaextent.width * 4 * cudaextent.depth)
            check_cudart_err(
                cudart.cudaMemcpy2DFromArray(self.new_cuda_mem_ptr, cudaextent.width * 4 * cudaextent.depth, array, 0,
                                             0,
                                             cudaextent.width * 4 * cudaextent.depth, cudaextent.height,
                                             cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice))
            self._cuda_buffer = cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(self.new_cuda_mem_ptr,
                                      cudaextent.width * cudaextent.depth * 4 * cudaextent.height,
                                      self), 0)
        return self.cuda_array

    def unmap(self, stream=None):
        if not self.registered:
            raise RuntimeError("Cannot unmap an unregistered buffer.")
        if not self.mapped:
            return self

        self._cuda_buffer = check_cudart_err(cudart.cudaGraphicsUnmapResources(1, self._graphics_resource, stream))

        return self

    def step(self):
        self.engine.taskMgr.step()
        if not self.registered and self.texture_context_future.done():
            self.texture_identifier = self.texture_context_future.result().getNativeId()
            self.register()


if __name__ == "__main__":
    # loadPrcFileData("", "threading-model Cull/Draw")
    #
    env = CUDATest(window_type="offscreen")
    env.step()
    env.step()

    for _ in range(10000000):
        env.step()
        with env as array:
            ret = array
            # np_array = cp.asnumpy(ret)
        pass
        # if ret is not None:
        #     np_ret = cp.asnumpy(ret)
        # if ret is not None:
        #     image = torch.as_tensor(ret, device='cpu')
        # img = np.frombuffer(env.my_texture.getRamImage().getData(), dtype=np.uint8)
        # img = img.reshape((env.my_texture.getYSize(), env.my_texture.getXSize(), 4))
        # img = img[::-1]
        # img = img[..., :-1]
        # cv2.imshow("window", img)
        # cv2.waitKey(1)
