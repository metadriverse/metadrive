"""
This file visualizes a single track.
Please draw the left button of mouse down to zoom out and see the whole picture of the scene.
"""
from metadrive.component.pg_space import Parameter
from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.roundabout import Roundabout
from metadrive.component.pgblock.std_intersection import StdInterSection
from metadrive.component.pgblock.straight import Straight
from metadrive.component.pgblock.t_intersection import TInterSection
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.engine.asset_loader import initialize_asset_loader
from metadrive.tests.vis_block.vis_block_base import TestBlock

if __name__ == "__main__":
    FirstPGBlock.ENTRANCE_LENGTH = 0.5
    test = TestBlock(False)

    initialize_asset_loader(engine=test)

    global_network = NodeRoadNetwork()
    blocks = []
    init_block = FirstPGBlock(global_network, 3.0, 3, test.render, test.world, 1)

    block_s1 = Straight(1, init_block.get_socket(0), global_network, 1)
    block_s1.construct_from_config(
        {
            Parameter.length: 100
        }, test.render, test.world
    )

    block_c1 = Curve(2, block_s1.get_socket(0), global_network, 1)
    block_c1.construct_from_config({
        Parameter.length: 200,
        Parameter.radius: 100,
        Parameter.angle: 90,
        Parameter.dir: 1,
    }, test.render, test.world)

    block_s2 = Straight(3, block_c1.get_socket(0), global_network, 1)
    block_s2.construct_from_config(
        {
            Parameter.length: 100,
        }, test.render, test.world
    )


    block_c2 = Curve(4, block_s2.get_socket(0), global_network, 1)
    block_c2.construct_from_config({
        Parameter.length: 100,
        Parameter.radius: 60,
        Parameter.angle: 90,
        Parameter.dir: 1,
    }, test.render, test.world)


    block_c3 = Curve(5, block_c2.get_socket(0), global_network, 1)
    block_c3.construct_from_config({
        Parameter.length: 100,
        Parameter.radius: 60,
        Parameter.angle: 90,
        Parameter.dir: 1,
    }, test.render, test.world)


    block_s3 = Straight(6, block_c3.get_socket(0), global_network, 1)
    block_s3.construct_from_config(
        {
            Parameter.length: 200,
        }, test.render, test.world
    )

    block_c4 = Curve(7, block_s3.get_socket(0), global_network, 1)
    block_c4.construct_from_config({
        Parameter.length: 100,
        Parameter.radius: 60,
        Parameter.angle: 90,
        Parameter.dir: 1,
    }, test.render, test.world)


    # block_c5 = Curve(8, block_c4.get_socket(0), global_network, 1)
    # block_c5.construct_from_config({
    #     Parameter.length: 60,
    #     Parameter.radius: 60,
    #     Parameter.angle: 90,
    #     Parameter.dir: 1,
    # }, test.render, test.world)




    test.run()
