"""This file visualizes a small town. Please zoom out in the pop-up window."""
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

    block = StdInterSection(1, init_block.get_socket(0), global_network, 1)
    block.construct_from_config(
        {
            Parameter.radius: 10,
            Parameter.change_lane_num: 0,
            Parameter.decrease_increase: 0
        }, test.render, test.world
    )

    block = Straight(2, block.get_socket(1), global_network, 1)
    block.construct_from_config({Parameter.length: 388}, test.render, test.world)

    block = StdInterSection(3, block.get_socket(0), global_network, 1)
    block.construct_from_config(
        {
            Parameter.radius: 10,
            Parameter.change_lane_num: 0,
            Parameter.decrease_increase: 0
        }, test.render, test.world
    )

    block = Straight(4, block.get_socket(2), global_network, 1)
    block.construct_from_config({Parameter.length: 70}, test.render, test.world)

    t_1 = TInterSection(5, block.get_socket(0), global_network, 1)
    t_1.construct_from_config(
        {
            Parameter.radius: 10,
            Parameter.change_lane_num: 1,
            Parameter.decrease_increase: 0,
            Parameter.t_intersection_type: 0
        }, test.render, test.world
    )

    block = Straight(6, t_1.get_socket(0), global_network, 1)
    block.construct_from_config({Parameter.length: 68}, test.render, test.world)

    t_last = TInterSection(100, block.get_socket(0), global_network, 1)
    t_last.construct_from_config(
        {
            Parameter.radius: 10,
            Parameter.change_lane_num: 2,
            Parameter.decrease_increase: 0,
            Parameter.t_intersection_type: 0
        }, test.render, test.world
    )

    block = Straight(101, t_last.get_socket(0), global_network, 1)
    block.construct_from_config({Parameter.length: 9}, test.render, test.world)

    t_2 = TInterSection(7, block.get_socket(0), global_network, 1)
    t_2.construct_from_config(
        {
            Parameter.radius: 10,
            Parameter.change_lane_num: 1,
            Parameter.decrease_increase: 0,
            Parameter.t_intersection_type: 0
        }, test.render, test.world
    )

    block = Straight(8, t_2.get_socket(0), global_network, 1)
    block.construct_from_config({Parameter.length: 70}, test.render, test.world)

    block = StdInterSection(9, block.get_socket(0), global_network, 1)
    block.construct_from_config(
        {
            Parameter.radius: 10,
            Parameter.change_lane_num: 0,
            Parameter.decrease_increase: 0
        }, test.render, test.world
    )

    block = Straight(10, block.get_socket(2), global_network, 1)
    block.construct_from_config({Parameter.length: 70}, test.render, test.world)

    t_3 = TInterSection(11, block.get_socket(0), global_network, 1)
    t_3.construct_from_config(
        {
            Parameter.radius: 10,
            Parameter.change_lane_num: 1,
            Parameter.decrease_increase: 0,
            Parameter.t_intersection_type: 0
        }, test.render, test.world
    )

    block = Straight(12, t_3.get_socket(0), global_network, 1)
    block.construct_from_config({Parameter.length: 130}, test.render, test.world)

    t_4 = TInterSection(13, block.get_socket(0), global_network, 1)
    t_4.construct_from_config(
        {
            Parameter.radius: 10,
            Parameter.change_lane_num: 1,
            Parameter.decrease_increase: 0,
            Parameter.t_intersection_type: 0
        }, test.render, test.world
    )

    block = Straight(14, t_4.get_socket(0), global_network, 1)
    block.construct_from_config({Parameter.length: 70}, test.render, test.world)

    block = StdInterSection(15, block.get_socket(0), global_network, 1)
    block.construct_from_config(
        {
            Parameter.radius: 10,
            Parameter.change_lane_num: 0,
            Parameter.decrease_increase: 0
        }, test.render, test.world
    )

    block = Straight(16, block.get_socket(2), global_network, 1)
    block.construct_from_config({Parameter.length: 259}, test.render, test.world)

    t_5 = TInterSection(17, block.get_socket(0), global_network, 1)
    t_5.construct_from_config(
        {
            Parameter.radius: 10,
            Parameter.change_lane_num: 1,
            Parameter.decrease_increase: 0,
            Parameter.t_intersection_type: 0
        }, test.render, test.world
    )

    block = Straight(18, t_5.get_socket(0), global_network, 1)
    block.construct_from_config({Parameter.length: 47}, test.render, test.world)

    block = Straight(19, t_5.get_socket(1), global_network, 1)
    block.construct_from_config({Parameter.length: 70}, test.render, test.world)

    t_6 = TInterSection(20, block.get_socket(0), global_network, 1)
    t_6.construct_from_config(
        {
            Parameter.radius: 10,
            Parameter.change_lane_num: 0,
            Parameter.decrease_increase: 0,
            Parameter.t_intersection_type: 0
        }, test.render, test.world
    )

    block = Straight(21, t_6.get_socket(1), global_network, 1)
    block.construct_from_config({Parameter.length: 130}, test.render, test.world)

    t_7 = TInterSection(22, block.get_socket(0), global_network, 1)
    t_7.construct_from_config(
        {
            Parameter.radius: 10,
            Parameter.change_lane_num: 0,
            Parameter.decrease_increase: 0,
            Parameter.t_intersection_type: 2
        }, test.render, test.world
    )

    block = Straight(23, t_7.get_socket(1), global_network, 1)
    block.construct_from_config({Parameter.length: 50}, test.render, test.world)

    block = Straight(24, t_7.get_socket(0), global_network, 1)
    block.construct_from_config({Parameter.length: 130}, test.render, test.world)

    t_8 = TInterSection(25, block.get_socket(0), global_network, 1)
    t_8.construct_from_config(
        {
            Parameter.radius: 10,
            Parameter.change_lane_num: 0,
            Parameter.decrease_increase: 0,
            Parameter.t_intersection_type: 0
        }, test.render, test.world
    )

    block = Straight(26, t_8.get_socket(1), global_network, 1)
    block.construct_from_config({Parameter.length: 50}, test.render, test.world)

    t_9 = TInterSection(27, t_8.get_socket(0), global_network, 1)
    t_9.construct_from_config(
        {
            Parameter.radius: 10,
            Parameter.change_lane_num: 1,
            Parameter.decrease_increase: 0,
            Parameter.t_intersection_type: 2
        }, test.render, test.world
    )

    o = Roundabout(28, t_9.get_socket(0), global_network, 1)
    o.construct_from_config(
        {
            Parameter.radius_exit: 5,
            Parameter.radius_inner: 30,
            Parameter.angle: 60
        }, test.render, test.world
    )

    curve = Curve(29, o.get_socket(0), global_network, 1)
    curve.construct_from_config(
        {
            Parameter.radius: 20,
            Parameter.dir: 0,
            Parameter.angle: 90,
            Parameter.length: 70
        }, test.render, test.world
    )

    block = Straight(30, t_6.get_socket(0), global_network, 1)
    block.construct_from_config({Parameter.length: 120}, test.render, test.world)

    t_10 = TInterSection(31, block.get_socket(0), global_network, 1)
    t_10.construct_from_config(
        {
            Parameter.radius: 10,
            Parameter.change_lane_num: 1,
            Parameter.decrease_increase: 0,
            Parameter.t_intersection_type: 0
        }, test.render, test.world
    )

    block = Straight(32, t_10.get_socket(0), global_network, 1)
    block.construct_from_config({Parameter.length: 16}, test.render, test.world)

    t_11 = TInterSection(33, block.get_socket(0), global_network, 1)
    t_11.construct_from_config(
        {
            Parameter.radius: 10,
            Parameter.change_lane_num: 1,
            Parameter.decrease_increase: 0,
            Parameter.t_intersection_type: 0
        }, test.render, test.world
    )

    block = Straight(34, t_11.get_socket(1), global_network, 1)
    block.construct_from_config({Parameter.length: 40}, test.render, test.world)

    test.run()
