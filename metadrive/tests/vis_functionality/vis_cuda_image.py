import time
import cv2
from panda3d.core import Texture, GraphicsOutput, GraphicsStateGuardianBase

import numpy as np

from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.intersection import InterSection
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.tests.vis_block.vis_block_base import TestBlock

if __name__ == "__main__":
    engine = TestBlock(window_type="offscreen")
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

    start_time = time.time()
    engine.taskMgr.step()
    engine.win.add_render_texture(my_texture, GraphicsOutput.RTMCopyRam)


    for i in range(10000):
        engine.taskMgr.step()
        origin_img = engine.win.getDisplayRegion(0).getScreenshot()
        img = np.frombuffer(origin_img.getRamImage().getData(), dtype=np.uint8)
        img = img.reshape((origin_img.getYSize(), origin_img.getXSize(), 4))
        img = img[::-1]
        img = img[..., :-1]

        # gsg = GraphicsStateGuardianBase.getDefaultGsg()
        # texture_context = my_texture.prepareNow(0, gsg.prepared_objects, gsg)
        # # texture_context = my_texture.prepare(0)
        # identifier = texture_context.getNativeId()

        # img = np.frombuffer(my_texture.getRamImage().getData(), dtype=np.uint8)
        # img = img.reshape((my_texture.getYSize(), my_texture.getXSize(), 4))
        # img = img[::-1]
        # img = img[..., :-1]
        # cv2.imshow("window", img)
        # cv2.waitKey(1)

        if i % 1000 == 0 and i != 0:
            print("FPS: {}".format(i / (time.time() - start_time)))
