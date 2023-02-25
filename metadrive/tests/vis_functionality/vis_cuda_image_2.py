import cupy as cp
import cv2
import numpy as np
from OpenGL.GL import GL_TEXTURE_2D, GL_TEXTURE_3D
from cuda.cudart import cudaGraphicsGLRegisterImage, cudaGraphicsRegisterFlags, cudaGraphicsSubResourceGetMappedArray, \
    cudaGraphicsMapResources, cudaArrayGetInfo, cudaMalloc, cudaArrayGetMemoryRequirements, cudaInitDevice
from panda3d.core import Texture, GraphicsOutput, GraphicsStateGuardianBase

from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.intersection import InterSection
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.tests.vis_block.vis_block_base import TestBlock
from panda3d.core import loadPrcFileData

# loadPrcFileData("", "win-size {} {}".format(1024, 1024))

# require:
# 1. pip install cupy-cuda12x
# 2. CUDA-Python
# 3. PyOpenGL
# 4. pyrr
# 5. glfw


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

        # set texture
        self.my_texture = Texture()
        # self.my_texture.setMinfilter(Texture.FTLinear)
        # self.my_texture.setFormat(Texture.FRgba32)
        engine.win.add_render_texture(self.my_texture, GraphicsOutput.RTMCopyRam)

        # get context future
        gsg = GraphicsStateGuardianBase.getDefaultGsg()
        self.texture_context_future = self.my_texture.prepare(gsg.prepared_objects)
        self.resource = None
        self.cp_array_pointer = None
        self.new_mem_ptr = None
        self.mapped_array = None

    def step(self):
        self.engine.taskMgr.step()
        if self.texture_context_future.done():
            # if self.resource is None:
            self.engine.graphicsEngine.renderFrame()
            self.engine.graphicsEngine.renderFrame()
            identifier = self.texture_context_future.result().getNativeId()
            flag, self.resource = cudaGraphicsGLRegisterImage(
                identifier, GL_TEXTURE_2D, cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsSurfaceLoadStore
            )
            map_success = cudaGraphicsMapResources(1, self.resource, 0)
            get_success, self.mapped_array = cudaGraphicsSubResourceGetMappedArray(self.resource, 0, 0)
            info_success, channelformat, cudaextent, flag = cudaArrayGetInfo(self.mapped_array)
            success, self.new_mem_ptr = cudaMalloc(cudaextent.height * cudaextent.width * 4 * 4)
            # ret = cudaMemcpy(self.new_mem_ptr, 1024 * 1024 * 4 * 4, self.mapped_array,
            #                  cudaMemcpyKind.cudaMemcpyDeviceToDevice)
            self.cp_array_pointer = cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(self.new_mem_ptr, cudaextent.height * cudaextent.width * 4 * 4, self), 0
            )
            # cudaMemcpy2DFromArray(new_mem_ptr, cudaextent.width * 4 * 4, array, 0, 0, cudaextent.width * 4 * 4, cudaextent.height, cudaMemcpyKind.cudaMemcpyDeviceToDevice)
            # assert ret
            return cp.ndarray(shape=(1024, 1024, 4), strides=None, order="C", memptr=self.cp_array_pointer)
        else:
            return None


if __name__ == "__main__":
    # loadPrcFileData("", "threading-model Cull/Draw")
    env = CUDATest(window_type="offscreen")
    for _ in range(10000000):
        ret = env.step()
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
