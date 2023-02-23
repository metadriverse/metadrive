import time
import cv2
from panda3d.core import Texture, GraphicsOutput, GraphicsStateGuardianBase, GraphicsStateGuardian, CallbackObject, \
    loadPrcFileData
from cuda.cudart import cudaGraphicsGLRegisterImage, cudaGraphicsRegisterFlags, GLuint, GLenum, \
    cudaGraphicsSubResourceGetMappedArray, cudaGraphicsMapResources, cudaArrayGetInfo, cudaMalloc, cudaMemcpy2DFromArray, cudaGraphicsResourceGetMappedPointer, cudaMemcpyKind, cudaMemcpy
import torch
from torch.utils.dlpack import from_dlpack

import numpy as np
import cupy as cp
from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.intersection import InterSection
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.tests.vis_block.vis_block_base import TestBlock
from OpenGL.GL import GL_TEXTURE_2D

# require:
# 1. pip install cupy-cuda12x
# 2. CUDA-Python
# 3. PyOpenGL
# 4. pyrr
# 5. glfw

class CUDATest:
    def __init__(self):
        self.engine = engine = TestBlock(window_type="onscreen")
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
        my_texture = Texture()
        my_texture.setMinfilter(Texture.FTLinear)
        my_texture.setFormat(Texture.FRgba32)
        type = my_texture.get_texture_type()
        engine.win.add_render_texture(my_texture, GraphicsOutput.RTMCopyTexture)

        # get context future
        gsg = GraphicsStateGuardianBase.getDefaultGsg()
        self.texture_context_future = my_texture.prepare(gsg.prepared_objects)
        self.resource = None

    def step(self):
        self.engine.taskMgr.step()
        if self.texture_context_future.done():
            self.engine.graphicsEngine.renderFrame()
            self.engine.graphicsEngine.renderFrame()
            identifier = self.texture_context_future.result().getNativeId()
            flag, self.resource = cudaGraphicsGLRegisterImage(identifier, GL_TEXTURE_2D,
                                                         cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly)
            map_success = cudaGraphicsMapResources(1,self.resource, 0)
            get_success, array = cudaGraphicsSubResourceGetMappedArray(self.resource, 0, 0)
            info_success, channelformat, cudaextent, flag = cudaArrayGetInfo(array)

            success, self.new_mem_ptr = cudaMalloc(cudaextent.height * cudaextent.width * 4 * 4)
            self.cp_array_pointer = cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(self.new_mem_ptr, cudaextent.height * cudaextent.width * 4 * 4, self), 0)
        else:
            return None
        # cudaMemcpy2DFromArray(new_mem_ptr, cudaextent.width * 4 * 4, array, 0, 0, cudaextent.width * 4 * 4, cudaextent.height, cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        ret=cudaMemcpy(self.new_mem_ptr, cudaextent.height*cudaextent.width*4*4, array, cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        return cp.ndarray(
            shape=(1024, 1024, 4),
            dtype=float,
            strides=None,
            order="C",
            memptr=self.cp_array_pointer,
        )


if __name__ == "__main__":
    # loadPrcFileData("", "threading-model Cull/Draw")
    env = CUDATest()
    for _ in range(10000000):
        ret = env.step()
        if ret is not None:
            image = torch.as_tensor(ret, device='cpu')