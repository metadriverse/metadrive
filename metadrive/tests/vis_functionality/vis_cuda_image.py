from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.intersection import InterSection
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.tests.vis_block.vis_block_base import TestBlock

if __name__ == "__main__":
    engine = TestBlock()
    from metadrive.engine.asset_loader import initialize_asset_loader
    initialize_asset_loader(engine)
    global_network = NodeRoadNetwork()
    first = FirstPGBlock(global_network, 3.0, 2, engine.render, engine.world, 20)
    intersection = InterSection(3, first.get_socket(0), global_network, 1)
    print(intersection.construct_block(engine.render, engine.world))
    engine.show_bounding_box(global_network)
    while True:
        engine = get_engine()
        assert engine.main_camera.current_track_vehicle is vehicle, "Tracked vehicle mismatch"
        if engine.episode_step <= 1:
            engine.graphicsEngine.renderFrame()
        origin_img = engine.win.getDisplayRegion(0).getScreenshot()
        v = memoryview(origin_img.getRamImage()).tolist()

        engine.taskMgr.step()
