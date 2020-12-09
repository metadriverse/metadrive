import logging
import os
from typing import Dict, Union, List

import numpy
from panda3d.bullet import BulletBoxShape, BulletRigidBodyNode, BulletWorld
from panda3d.core import Vec3, LQuaternionf, BitMask32, NodePath, Vec4, CardMaker, TextureStage, RigidBodyCombiner, \
    TransparencyAttrib, SamplerState

from pg_drive.pg_config.body_name import BodyName
from pg_drive.pg_config.cam_mask import CamMask
from pg_drive.scene_creator.lanes.circular_lane import CircularLane
from pg_drive.scene_creator.lanes.lane import AbstractLane, LineType
from pg_drive.scene_creator.lanes.straight_lane import StraightLane
from pg_drive.scene_creator.road.road import Road
from pg_drive.scene_creator.road.road_network import RoadNetwork
from pg_drive.utils.element import Element
from pg_drive.utils.math_utils import norm
from pg_drive.utils.asset_loader import AssetLoader


class BlockSocket:
    """
    A pair of roads in reverse direction
    Positive_road is right road, and Negative road is left road on which cars drive in reverse direction
    BlockSocket is a part of block used to connect other blocks
    """
    def __init__(self, positive_road: Road, negative_road: Road = None):
        self.positive_road = positive_road
        self.negative_road = negative_road if negative_road else None
        self.index = None


class Block(Element):
    """
    Abstract class of Block,
    BlockSocket: a part of previous block connecting this block

    <----------------------------------------------
    road_2_end <---------------------- road_2_start
    <~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>
    road_1_start ----------------------> road_1_end
    ---------------------------------------------->
    BlockSocket = tuple(road_1, road_2)

    When single-direction block created, road_2 in block socket is useless.
    But it's helpful when a town is created.
    """
    CENTER_LINE_TYPE = LineType.CONTINUOUS

    # road network property
    ID = None  # each block must have a unique ID
    SOCKET_NUM = None

    # visualization size property
    CIRCULAR_SEGMENT_LENGTH = 4
    STRIPE_LENGTH = 1.5
    LANE_LINE_WIDTH = 0.15
    LANE_LINE_THICKNESS = 0.01

    SIDE_WALK_THICKNESS = 0.4
    SIDE_WALK_LENGTH = 3
    SIDE_WALK_WIDTH = 3
    SIDE_WALK_LINE_DIST = 0.6

    # visualization color property
    LAND_COLOR = (0.4, 0.4, 0.4, 1)
    NAVI_COLOR = (0.709, 0.09, 0, 1)

    # lane line collision group
    COLLISION_MASK = 3

    # for creating complex block, for example Intersection and roundabout consist of 4 part, which contain several road
    PART_IDX = 0
    ROAD_IDX = 0
    DASH = "_"

    def __init__(self, block_index: int, pre_block_socket: BlockSocket, global_network: RoadNetwork, random_seed):
        super(Block, self).__init__(random_seed)
        # block information
        assert self.ID is not None, "Each Block must has its unique ID When define Block"
        assert self.SOCKET_NUM is not None, "The number of Socket should be specified when define a new block"
        if block_index == 0:
            from pg_drive.scene_creator.blocks.first_block import FirstBlock
            assert isinstance(self, FirstBlock), "only first block can use block index 0"
        elif block_index < 0:
            logging.debug("It is recommended that block index should > 1")
        self._block_name = str(block_index) + self.ID
        self.block_index = block_index
        self.number_of_sample_trial = 0

        # each block contains its own road network and a global network
        self._global_network = global_network
        self.block_network = RoadNetwork()

        # used to spawn npc
        self._reborn_roads = []

        # own sockets, one block derives from a socket, but will have more sockets to connect other blocks
        self._sockets = []

        # used to connect previous blocks, save its info here
        self._pre_block_socket = pre_block_socket
        self.pre_block_socket_index = pre_block_socket.index

        # used to create this block, but for first block it is nonsense
        if block_index != 0:
            self.positive_lanes = self._pre_block_socket.positive_road.get_lanes(self._global_network)
            self.negative_lanes = self._pre_block_socket.negative_road.get_lanes(self._global_network)
            self.positive_lane_num = len(self.positive_lanes)
            self.negative_lane_num = len(self.negative_lanes)
            self.positive_basic_lane = self.positive_lanes[-1]  # most right or outside lane is the basic lane
            self.negative_basic_lane = self.negative_lanes[-1]  # most right or outside lane is the basic lane
            self.lane_width = self.positive_basic_lane.width_at(0)

        if self.render:
            # render pre-load
            self.road_texture = self.loader.loadTexture(
                AssetLoader.file_path(AssetLoader.asset_path, "textures", "sci", "color.jpg")
            )
            self.road_texture.setMinfilter(SamplerState.FT_linear_mipmap_linear)
            self.road_texture.setAnisotropicDegree(8)
            self.road_normal = self.loader.loadTexture(
                AssetLoader.file_path(AssetLoader.asset_path, "textures", "sci", "normal.jpg")
            )
            self.ts_color = TextureStage("color")
            self.ts_normal = TextureStage("normal")
            self.side_texture = self.loader.loadTexture(
                AssetLoader.file_path(AssetLoader.asset_path, "textures", "side_walk", "color.png")
            )
            self.side_texture.setMinfilter(SamplerState.FT_linear_mipmap_linear)
            self.side_texture.setAnisotropicDegree(8)
            self.side_normal = self.loader.loadTexture(
                AssetLoader.file_path(AssetLoader.asset_path, "textures", "side_walk", "normal.png")
            )
            self.side_walk = self.loader.loadModel(AssetLoader.file_path(AssetLoader.asset_path, "models", "box.bam"))

    def construct_block_random(self, root_render_np: NodePath, pg_physics_world: BulletWorld) -> bool:
        self.set_config(self.PARAMETER_SPACE.sample())
        success = self._sample_topology()
        self._create_in_world()
        self.attach_to_pg_world(root_render_np, pg_physics_world)
        return success

    def destruct_block(self, pg_physics_world: BulletWorld):
        self._clear_topology()
        if len(self.bullet_nodes) != 0:
            for node in self.bullet_nodes:
                pg_physics_world.remove(node)
            self.bullet_nodes.clear()
        if self.node_path is not None:
            self.node_path.removeNode()
            self.node_path = None

    def _sample_topology(self) -> bool:
        """
        Sample a new topology, clear the previous settings at first
        """
        self.number_of_sample_trial += 1
        self._clear_topology()
        no_cross = self._try_plug_into_previous_block()
        for i, s in enumerate(self._sockets):
            s.index = i
        self._global_network += self.block_network
        return no_cross

    def construct_from_config(self, config: Dict, root_render_np: NodePath, pg_physics_world: BulletWorld):
        assert set(config.keys()) == self.PARAMETER_SPACE.parameters, \
            "Make sure the parameters' name are as same as what defined in parameter_space.py"
        self.set_config(config)
        success = self._sample_topology()
        self._create_in_world()
        self.attach_to_pg_world(root_render_np, pg_physics_world)
        return success

    def get_socket(self, index: int) -> BlockSocket:
        """
        Get i th socket
        """
        if index < 0 or index >= len(self._sockets):
            raise ValueError("Socket of {}: index out of range".format(self.class_name))
        return self._sockets[index]

    def add_reborn_roads(self, reborn_roads: Union[List[Road], Road]):
        """
        Use this to add spawn roads instead of modifying the list directly
        """
        if isinstance(reborn_roads, List):
            for road in reborn_roads:
                self._add_one_reborn_road(road)
        elif isinstance(reborn_roads, Road):
            self._add_one_reborn_road(reborn_roads)
        else:
            raise ValueError("Only accept List[Road] or Road in this func")

    def get_reborn_roads(self):
        return self._reborn_roads

    def get_reborn_lanes(self):
        """
        return a 2-dim array [[]] to keep the lane index
        """
        ret = []
        for road in self._reborn_roads:
            lanes = road.get_lanes(self.block_network)
            ret.append(lanes)
        return ret

    def add_sockets(self, sockets: Union[List[BlockSocket], BlockSocket]):
        """
        Use this to add sockets instead of modifying the list directly
        """
        if isinstance(sockets, BlockSocket):
            self._add_one_socket(sockets)
        elif isinstance(sockets, List):
            for socket in sockets:
                self._add_one_socket(socket)

    def set_part_idx(self, x):
        """
        It is necessary to divide block to some parts in complex block and give them unique id according to part idx
        """
        self.PART_IDX = x
        self.ROAD_IDX = 0  # clear the road idx when create new part

    def add_road_node(self):
        """
        Call me to get a new node name of this block.
        It is more accurate and recommended to use road_node() to get a node name
        """
        self.ROAD_IDX += 1
        return self.road_node(self.PART_IDX, self.ROAD_IDX - 1)

    def road_node(self, part_idx: int, road_idx: int) -> str:
        """
        return standard road node name
        """
        return self._block_name + str(part_idx) + self.DASH + str(road_idx) + self.DASH

    def _add_one_socket(self, socket: BlockSocket):
        assert isinstance(socket, BlockSocket), "Socket list only accept BlockSocket Type"
        self._sockets.append(socket)

    def _add_one_reborn_road(self, reborn_road: Road):
        assert isinstance(reborn_road, Road), "Spawn roads list only accept Road Type"
        self._reborn_roads.append(reborn_road)

    def _clear_topology(self):
        self._global_network -= self.block_network
        self.block_network.graph.clear()
        self.PART_IDX = 0
        self.ROAD_IDX = 0
        self._reborn_roads.clear()
        self._sockets.clear()

    def _try_plug_into_previous_block(self) -> bool:
        """
        Try to plug this Block to previous block's socket, return True for success, False for road cross
        """
        raise NotImplementedError

    """------------------------------------- For Render and Physics Calculation ---------------------------------- """

    def _create_in_world(self):
        """
        Create NodePath and Geom node to perform both collision detection and render
        """
        self.model_node_path = NodePath(RigidBodyCombiner(self._block_name + "_model"))
        self.card_node_path = NodePath(RigidBodyCombiner(self._block_name + "_card"))
        graph = self.block_network.graph
        for _from, to_dict in graph.items():
            for _to, lanes in to_dict.items():
                if self.render:
                    self._add_land(lanes)
                for _id, l in enumerate(lanes):
                    line_color = l.line_color
                    self._add_lane(l, _id, line_color)
        self.model_node_path.flattenStrong()
        self.model_node_path.node().collect()

        self.card_node_path.flattenStrong()
        self.card_node_path.node().collect()
        self.card_node_path.hide(CamMask.DepthCam)

        self.node_path = NodePath(self._block_name)
        self.node_path.hide(CamMask.Shadow)
        self.model_node_path.reparentTo(self.node_path)
        self.card_node_path.reparentTo(self.node_path)

    def _add_lane(self, lane: AbstractLane, lane_id: int, colors: List[Vec4]):
        parent_np = self.model_node_path
        lane_width = lane.width_at(0)
        for k, i in enumerate([-1, 1]):
            line_color = colors[k]
            if lane.line_types[k] == LineType.NONE or (lane_id != 0 and k == 0):
                if isinstance(lane, StraightLane):
                    continue
                elif isinstance(lane, CircularLane) and lane.radius != lane_width / 2:
                    # for ramp render
                    continue
            if lane.line_types[k] == LineType.CONTINUOUS or lane.line_types[k] == LineType.SIDE:
                if isinstance(lane, StraightLane):
                    lane_start = lane.position(0, i * lane_width / 2)
                    lane_end = lane.position(lane.length, i * lane_width / 2)
                    middle = lane.position(lane.length / 2, i * lane_width / 2)
                    self._add_lane_line2bullet(lane_start, lane_end, middle, parent_np, line_color, lane.line_types[k])
                elif isinstance(lane, CircularLane):
                    segment_num = int(lane.length / Block.CIRCULAR_SEGMENT_LENGTH)
                    for segment in range(segment_num):
                        lane_start = lane.position(segment * Block.CIRCULAR_SEGMENT_LENGTH, i * lane_width / 2)
                        lane_end = lane.position((segment + 1) * Block.CIRCULAR_SEGMENT_LENGTH, i * lane_width / 2)
                        middle = (lane_start + lane_end) / 2

                        self._add_lane_line2bullet(
                            lane_start, lane_end, middle, parent_np, line_color, lane.line_types[k]
                        )
                    # for last part
                    lane_start = lane.position(segment_num * Block.CIRCULAR_SEGMENT_LENGTH, i * lane_width / 2)
                    lane_end = lane.position(lane.length, i * lane_width / 2)
                    middle = (lane_start + lane_end) / 2
                    self._add_lane_line2bullet(lane_start, lane_end, middle, parent_np, line_color, lane.line_types[k])

                if lane.line_types[k] == LineType.SIDE:
                    radius = lane.radius if isinstance(lane, CircularLane) else 0.0
                    segment_num = int(lane.length / Block.SIDE_WALK_LENGTH)
                    for segment in range(segment_num):
                        lane_start = lane.position(segment * Block.SIDE_WALK_LENGTH, i * lane_width / 2)
                        lane_end = lane.position((segment + 1) * Block.SIDE_WALK_LENGTH, i * lane_width / 2)
                        middle = (lane_start + lane_end) / 2
                        self._add_side_walk2bullet(lane_start, lane_end, middle, radius, lane.direction)
                    # for last part
                    lane_start = lane.position(segment_num * Block.SIDE_WALK_LENGTH, i * lane_width / 2)
                    lane_end = lane.position(lane.length, i * lane_width / 2)
                    middle = (lane_start + lane_end) / 2
                    if norm(lane_start[0] - lane_end[0], lane_start[1] - lane_end[1]) > 1e-1:
                        self._add_side_walk2bullet(lane_start, lane_end, middle, radius, lane.direction)

            elif lane.line_types[k] == LineType.STRIPED:
                straight = True if isinstance(lane, StraightLane) else False
                segment_num = int(lane.length / (2 * Block.STRIPE_LENGTH))
                for segment in range(segment_num):
                    lane_start = lane.position(segment * Block.STRIPE_LENGTH * 2, i * lane_width / 2)
                    lane_end = lane.position(
                        segment * Block.STRIPE_LENGTH * 2 + Block.STRIPE_LENGTH, i * lane_width / 2
                    )
                    middle = lane.position(
                        segment * Block.STRIPE_LENGTH * 2 + Block.STRIPE_LENGTH / 2, i * lane_width / 2
                    )

                    self._add_lane_line2bullet(
                        lane_start, lane_end, middle, parent_np, line_color, lane.line_types[k], straight
                    )

                if straight:
                    lane_start = lane.position(0, i * lane_width / 2)
                    lane_end = lane.position(lane.length, i * lane_width / 2)
                    middle = lane.position(lane.length / 2, i * lane_width / 2)
                    self._add_box_body(lane_start, lane_end, middle, parent_np, lane.line_types[k])

    def _add_box_body(self, lane_start, lane_end, middle, parent_np: NodePath, line_type):
        length = norm(lane_end[0] - lane_start[0], lane_end[1] - lane_start[1])
        if LineType.prohibit(line_type):
            node_name = BodyName.Continuous_line
        else:
            node_name = BodyName.Stripped_line
        body_node = BulletRigidBodyNode(node_name)
        body_node.setActive(False)
        body_node.setKinematic(False)
        body_node.setStatic(True)
        body_np = parent_np.attachNewNode(body_node)
        shape = BulletBoxShape(Vec3(length / 2, Block.LANE_LINE_WIDTH / 2, Block.LANE_LINE_THICKNESS))
        body_np.node().addShape(shape)
        body_np.node().setIntoCollideMask(BitMask32.bit(Block.COLLISION_MASK))
        self.bullet_nodes.append(body_np.node())

        body_np.setPos(middle[0], -middle[1], 0)
        direction_v = lane_end - lane_start
        theta = -numpy.arctan2(direction_v[1], direction_v[0])
        body_np.setQuat(LQuaternionf(numpy.cos(theta / 2), 0, 0, numpy.sin(theta / 2)))

    def _add_lane_line2bullet(
        self,
        lane_start,
        lane_end,
        middle,
        parent_np: NodePath,
        color: Vec4,
        line_type: LineType,
        straight_stripe=False
    ):
        length = norm(lane_end[0] - lane_start[0], lane_end[1] - lane_start[1])
        if length <= 0:
            return
        if LineType.prohibit(line_type):
            node_name = BodyName.Continuous_line
        else:
            node_name = BodyName.Stripped_line

        # add bullet body for it
        if straight_stripe:
            body_np = parent_np.attachNewNode(node_name)
        else:
            scale = 2 if line_type == LineType.STRIPED else 1
            body_node = BulletRigidBodyNode(node_name)
            body_node.setActive(False)
            body_node.setKinematic(False)
            body_node.setStatic(True)
            body_np = parent_np.attachNewNode(body_node)
            shape = BulletBoxShape(Vec3(scale / 2, Block.LANE_LINE_WIDTH / 2, Block.LANE_LINE_THICKNESS))
            body_np.node().addShape(shape)
            body_np.node().setIntoCollideMask(BitMask32.bit(Block.COLLISION_MASK))
            self.bullet_nodes.append(body_np.node())

        # position and heading
        body_np.setPos(middle[0], -middle[1], 0)
        direction_v = lane_end - lane_start
        theta = -numpy.arctan2(direction_v[1], direction_v[0])
        body_np.setQuat(LQuaternionf(numpy.cos(theta / 2), 0, 0, numpy.sin(theta / 2)))

        if self.render:
            # For visualization
            lane_line = self.loader.loadModel(AssetLoader.file_path(AssetLoader.asset_path, "models", "box.bam"))
            lane_line.getChildren().reparentTo(body_np)
        body_np.setScale(length, Block.LANE_LINE_WIDTH, Block.LANE_LINE_THICKNESS)
        body_np.set_color(color)

    def _add_side_walk2bullet(self, lane_start, lane_end, middle, radius=0.0, direction=0):
        length = norm(lane_end[0] - lane_start[0], lane_end[1] - lane_start[1])
        body_node = BulletRigidBodyNode(BodyName.Side_walk)
        body_node.setActive(False)
        body_node.setKinematic(False)
        body_node.setStatic(True)
        side_np = self.model_node_path.attachNewNode(body_node)
        shape = BulletBoxShape(Vec3(1 / 2, 1 / 2, 1 / 2))
        body_node.addShape(shape)
        body_node.setIntoCollideMask(BitMask32.bit(Block.COLLISION_MASK))
        self.bullet_nodes.append(body_node)

        if radius == 0:
            factor = 1
        else:
            if direction == 1:
                factor = (1 - self.SIDE_WALK_LINE_DIST / radius)
            else:
                factor = (1 + self.SIDE_WALK_WIDTH / radius) * (1 + self.SIDE_WALK_LINE_DIST / radius)
        direction_v = lane_end - lane_start
        vertical_v = (-direction_v[1], direction_v[0]) / numpy.linalg.norm(direction_v)
        middle += vertical_v * (self.SIDE_WALK_WIDTH / 2 + self.SIDE_WALK_LINE_DIST)
        side_np.setPos(middle[0], -middle[1], 0)
        theta = -numpy.arctan2(direction_v[1], direction_v[0])
        side_np.setQuat(LQuaternionf(numpy.cos(theta / 2), 0, 0, numpy.sin(theta / 2)))
        side_np.setScale(
            length * factor, self.SIDE_WALK_WIDTH, self.SIDE_WALK_THICKNESS * (1 + 0.1 * numpy.random.rand())
        )
        if self.render:
            side_np.setTexture(self.ts_color, self.side_texture)
            self.side_walk.instanceTo(side_np)

    def _add_land(self, lanes):
        if isinstance(lanes[0], StraightLane):
            for lane in lanes:
                middle = lane.position(lane.length / 2, 0)
                end = lane.position(lane.length, 0)
                direction_v = end - middle
                theta = -numpy.arctan2(direction_v[1], direction_v[0])
                width = lane.width_at(0) + self.SIDE_WALK_LINE_DIST * 2
                length = lane.length
                self._add_land2bullet(middle, width, length, theta)
        else:
            for lane in lanes:
                segment_num = int(lane.length / self.CIRCULAR_SEGMENT_LENGTH)
                for i in range(segment_num):
                    middle = lane.position(lane.length * (i + .5) / segment_num, 0)
                    end = lane.position(lane.length * (i + 1) / segment_num, 0)
                    direction_v = end - middle
                    theta = -numpy.arctan2(direction_v[1], direction_v[0])
                    width = lane.width_at(0) + self.SIDE_WALK_LINE_DIST * 2
                    length = lane.length
                    self._add_land2bullet(middle, width, length * 1.3 / segment_num, theta)

    def _add_land2bullet(self, middle, width, length, theta):
        cm = CardMaker('card')
        cm.setFrame(-length / 2, length / 2, -width / 2, width / 2)
        cm.setHasNormals(True)
        cm.setUvRange((0, 0), (length / 20, width / 10))
        card = self.card_node_path.attachNewNode(cm.generate())
        card.setPos(middle[0], -middle[1], numpy.random.rand() * 0.01 - 0.01)

        card.setQuat(
            LQuaternionf(
                numpy.cos(theta / 2) * numpy.cos(-numpy.pi / 4),
                numpy.cos(theta / 2) * numpy.sin(-numpy.pi / 4), -numpy.sin(theta / 2) * numpy.cos(-numpy.pi / 4),
                numpy.sin(theta / 2) * numpy.cos(-numpy.pi / 4)
            )
        )
        card.setTransparency(TransparencyAttrib.MMultisample)
        card.setTexture(self.ts_color, self.road_texture)

    @staticmethod
    def create_socket_from_positive_road(road: Road) -> BlockSocket:
        """
        We usually create road from positive road, thus this func can get socket easily.
        Note: it is not recommended to generate socket from negative road
        """
        assert road.start_node[0] != Road.NEGATIVE_DIR and road.end_node[0] != Road.NEGATIVE_DIR, \
            "Socket can only be created from positive road"
        positive_road = Road(road.start_node, road.end_node)
        return BlockSocket(positive_road, -positive_road)
