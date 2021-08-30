import math
from typing import Dict, Union, List

import numpy as np
from panda3d.bullet import BulletBoxShape, BulletRigidBodyNode, BulletGhostNode
from panda3d.core import Vec3, LQuaternionf, Vec4, CardMaker, TextureStage, RigidBodyCombiner, \
    TransparencyAttrib, SamplerState, NodePath
from metadrive.base_class.base_object import BaseObject
from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.lane.circular_lane import CircularLane
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.lane.waypoint_lane import WayPointLane
from metadrive.component.road.road import Road
from metadrive.component.road.road_network import RoadNetwork
from metadrive.constants import BodyName, CamMask, LineType, LineColor, DrivableAreaProperty
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.core.physics_world import PhysicsWorld
from metadrive.engine.physics_node import BaseRigidBodyNode
from metadrive.utils.coordinates_shift import panda_position
from metadrive.utils.math_utils import norm, Vector


class BaseBlock(BaseObject, DrivableAreaProperty):
    """
    Block is a driving area consisting of several roads
    Note: overriding the _sample() function to fill block_network/respawn_roads in subclass
    Call Block.construct_block() to add it to world
    """

    ID = "B"

    def __init__(self, block_index: int, global_network: RoadNetwork, random_seed, ignore_intersection_checking=False):
        super(BaseBlock, self).__init__(str(block_index) + self.ID, random_seed, escape_random_seed_assertion=True)
        # block information
        assert self.ID is not None, "Each Block must has its unique ID When define Block"
        assert len(self.ID) == 1, "Block ID must be a character "

        self.block_index = block_index
        self.ignore_intersection_checking = ignore_intersection_checking

        # each block contains its own road network and a global network
        self._global_network = global_network
        self.block_network = RoadNetwork()

        # a bounding box used to improve efficiency x_min, x_max, y_min, y_max
        self.bounding_box = None

        # used to spawn npc
        self._respawn_roads = []
        self._block_objects = None

        if self.render:
            # render pre-load
            self.road_texture = self.loader.loadTexture(AssetLoader.file_path("textures", "sci", "color.jpg"))
            self.road_texture.setMinfilter(SamplerState.FT_linear_mipmap_linear)
            self.road_texture.setAnisotropicDegree(8)
            self.road_normal = self.loader.loadTexture(AssetLoader.file_path("textures", "sci", "normal.jpg"))
            self.ts_color = TextureStage("color")
            self.ts_normal = TextureStage("normal")
            self.side_texture = self.loader.loadTexture(AssetLoader.file_path("textures", "sidewalk", "color.png"))
            self.side_texture.setMinfilter(SamplerState.FT_linear_mipmap_linear)
            self.side_texture.setAnisotropicDegree(8)
            self.side_normal = self.loader.loadTexture(AssetLoader.file_path("textures", "sidewalk", "normal.png"))
            self.sidewalk = self.loader.loadModel(AssetLoader.file_path("models", "box.bam"))

    def _sample_topology(self) -> bool:
        """
        Sample a new topology to fill self.block_network
        """
        raise NotImplementedError

    def construct_block(
        self,
        root_render_np: NodePath,
        physics_world: PhysicsWorld,
        extra_config: Dict = None,
        no_same_node=True
    ) -> bool:
        """
        Randomly Construct a block, if overlap return False
        """
        self.sample_parameters()
        self.origin = NodePath(self.name)
        self._block_objects = []
        if extra_config:
            assert set(extra_config.keys()).issubset(self.PARAMETER_SPACE.parameters), \
                "Make sure the parameters' name are as same as what defined in pg_space.py"
            raw_config = self.get_config()
            raw_config.update(extra_config)
            self.update_config(raw_config)
        self._clear_topology()
        success = self._sample_topology()
        self._global_network.add(self.block_network, no_same_node)
        self._create_in_world()
        self.attach_to_world(root_render_np, physics_world)
        return success

    def destruct_block(self, physics_world: PhysicsWorld):
        self._clear_topology()
        self.detach_from_world(physics_world)
        self.origin.removeNode()
        self.dynamic_nodes.clear()
        self.static_nodes.clear()
        for obj in self._block_objects:
            obj.destroy()
        self._block_objects = None

    def construct_from_config(self, config: Dict, root_render_np: NodePath, physics_world: PhysicsWorld):
        success = self.construct_block(root_render_np, physics_world, config)
        return success

    def get_respawn_roads(self):
        return self._respawn_roads

    def get_respawn_lanes(self):
        """
        return a 2-dim array [[]] to keep the lane index
        """
        ret = []
        for road in self._respawn_roads:
            lanes = road.get_lanes(self.block_network)
            ret.append(lanes)
        return ret

    def get_intermediate_spawn_lanes(self):
        """Return all lanes that can be used to generate spawn intermediate vehicles."""
        raise NotImplementedError()

    def _add_one_respawn_road(self, respawn_road: Road):
        assert isinstance(respawn_road, Road), "Spawn roads list only accept Road Type"
        self._respawn_roads.append(respawn_road)

    def _clear_topology(self):
        self._global_network -= self.block_network
        self.block_network.graph.clear()
        self.PART_IDX = 0
        self.ROAD_IDX = 0
        self._respawn_roads.clear()

    """------------------------------------- For Render and Physics Calculation ---------------------------------- """

    def _create_in_world(self):
        """
        Create NodePath and Geom node to perform both collision detection and render
        """
        self.lane_line_node_path = NodePath(RigidBodyCombiner(self.name + "_lane_line"))
        self.sidewalk_node_path = NodePath(RigidBodyCombiner(self.name + "_sidewalk"))
        self.lane_node_path = NodePath(RigidBodyCombiner(self.name + "_lane"))
        self.lane_vis_node_path = NodePath(RigidBodyCombiner(self.name + "_lane_vis"))
        graph = self.block_network.graph
        for _from, to_dict in graph.items():
            for _to, lanes in to_dict.items():
                self._add_lane_surface(_from, _to, lanes)
                for _id, l in enumerate(lanes):
                    line_color = l.line_color
                    self._add_lane(l, _id, line_color)
        self.lane_line_node_path.flattenStrong()
        self.lane_line_node_path.node().collect()

        self.sidewalk_node_path.flattenStrong()
        self.sidewalk_node_path.node().collect()
        self.sidewalk_node_path.hide(CamMask.ScreenshotCam)

        # only bodies reparent to this node
        self.lane_node_path.flattenStrong()
        self.lane_node_path.node().collect()

        self.lane_vis_node_path.flattenStrong()
        self.lane_vis_node_path.node().collect()
        self.lane_vis_node_path.hide(CamMask.DepthCam | CamMask.ScreenshotCam)

        self.origin.hide(CamMask.Shadow)

        self.sidewalk_node_path.reparentTo(self.origin)
        self.lane_line_node_path.reparentTo(self.origin)
        self.lane_node_path.reparentTo(self.origin)
        self.lane_vis_node_path.reparentTo(self.origin)

        self.bounding_box = self.block_network.get_bounding_box()

    def _add_metadrive_lanes(self, lane, lane_id, lane_width, colors, parent_np):
        # for metadrive structure
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
                    segment_num = int(lane.length / DrivableAreaProperty.CIRCULAR_SEGMENT_LENGTH)
                    for segment in range(segment_num):
                        lane_start = lane.position(
                            segment * DrivableAreaProperty.CIRCULAR_SEGMENT_LENGTH, i * lane_width / 2
                        )
                        lane_end = lane.position(
                            (segment + 1) * DrivableAreaProperty.CIRCULAR_SEGMENT_LENGTH, i * lane_width / 2
                        )
                        middle = (lane_start + lane_end) / 2

                        self._add_lane_line2bullet(
                            lane_start, lane_end, middle, parent_np, line_color, lane.line_types[k]
                        )
                    # for last part
                    lane_start = lane.position(
                        segment_num * DrivableAreaProperty.CIRCULAR_SEGMENT_LENGTH, i * lane_width / 2
                    )
                    lane_end = lane.position(lane.length, i * lane_width / 2)
                    middle = (lane_start + lane_end) / 2
                    self._add_lane_line2bullet(lane_start, lane_end, middle, parent_np, line_color, lane.line_types[k])

                if lane.line_types[k] == LineType.SIDE:
                    radius = lane.radius if isinstance(lane, CircularLane) else 0.0
                    segment_num = int(lane.length / DrivableAreaProperty.SIDEWALK_LENGTH)
                    for segment in range(segment_num):
                        lane_start = lane.position(segment * DrivableAreaProperty.SIDEWALK_LENGTH, i * lane_width / 2)
                        lane_end = lane.position(
                            (segment + 1) * DrivableAreaProperty.SIDEWALK_LENGTH, i * lane_width / 2
                        )
                        middle = (lane_start + lane_end) / 2
                        self._add_sidewalk2bullet(lane_start, lane_end, middle, radius, lane.direction)
                    # for last part
                    lane_start = lane.position(segment_num * DrivableAreaProperty.SIDEWALK_LENGTH, i * lane_width / 2)
                    lane_end = lane.position(lane.length, i * lane_width / 2)
                    middle = (lane_start + lane_end) / 2
                    if norm(lane_start[0] - lane_end[0], lane_start[1] - lane_end[1]) > 1e-1:
                        self._add_sidewalk2bullet(lane_start, lane_end, middle, radius, lane.direction)

            elif lane.line_types[k] == LineType.BROKEN:
                straight = True if isinstance(lane, StraightLane) else False
                segment_num = int(lane.length / (2 * DrivableAreaProperty.STRIPE_LENGTH))
                for segment in range(segment_num):
                    lane_start = lane.position(segment * DrivableAreaProperty.STRIPE_LENGTH * 2, i * lane_width / 2)
                    lane_end = lane.position(
                        segment * DrivableAreaProperty.STRIPE_LENGTH * 2 + DrivableAreaProperty.STRIPE_LENGTH,
                        i * lane_width / 2
                    )
                    middle = lane.position(
                        segment * DrivableAreaProperty.STRIPE_LENGTH * 2 + DrivableAreaProperty.STRIPE_LENGTH / 2,
                        i * lane_width / 2
                    )

                    self._add_lane_line2bullet(
                        lane_start, lane_end, middle, parent_np, line_color, lane.line_types[k], straight
                    )

                lane_start = lane.position(segment_num * DrivableAreaProperty.STRIPE_LENGTH * 2, i * lane_width / 2)
                lane_end = lane.position(lane.length + DrivableAreaProperty.STRIPE_LENGTH, i * lane_width / 2)
                middle = (lane_end[0] + lane_start[0]) / 2, (lane_end[1] + lane_start[1]) / 2
                if not straight:
                    self._add_lane_line2bullet(
                        lane_start, lane_end, middle, parent_np, line_color, lane.line_types[k], straight
                    )
                if straight:
                    lane_start = lane.position(0, i * lane_width / 2)
                    lane_end = lane.position(lane.length, i * lane_width / 2)
                    middle = lane.position(lane.length / 2, i * lane_width / 2)
                    self._add_box_body(lane_start, lane_end, middle, parent_np, lane.line_types[k], line_color)

    def _add_lane(self, lane: AbstractLane, lane_id: int, colors: List[Vec4]):
        parent_np = self.lane_line_node_path
        lane_width = lane.width_at(0)
        if isinstance(lane, CircularLane) or isinstance(lane, StraightLane):
            self._add_metadrive_lanes(lane, lane_id, lane_width, colors, parent_np)
        elif isinstance(lane, WayPointLane):
            for c, i in enumerate([-1, 1]):
                line_color = colors[c]
                acc_length = 0
                if lane.line_types[c] != LineType.NONE:
                    for segment in lane.segment_property:
                        lane_start = lane.position(acc_length, i * lane_width / 2)
                        acc_length += segment["length"]
                        lane_end = lane.position(acc_length, i * lane_width / 2)
                        middle = (lane_start + lane_end) / 2
                        self._add_lane_line2bullet(
                            lane_start, lane_end, middle, parent_np, line_color, lane.line_types[c]
                        )

    def _add_box_body(self, lane_start, lane_end, middle, parent_np: NodePath, line_type, line_color):
        length = norm(lane_end[0] - lane_start[0], lane_end[1] - lane_start[1])
        if LineType.prohibit(line_type):
            node_name = BodyName.White_continuous_line if line_color == LineColor.GREY else BodyName.Yellow_continuous_line
        else:
            node_name = BodyName.Broken_line
        body_node = BulletGhostNode(node_name)
        body_node.set_active(False)
        body_node.setKinematic(False)
        body_node.setStatic(True)
        body_np = parent_np.attachNewNode(body_node)
        shape = BulletBoxShape(
            Vec3(length / 2, DrivableAreaProperty.LANE_LINE_WIDTH / 2, DrivableAreaProperty.LANE_LINE_GHOST_HEIGHT)
        )
        body_np.node().addShape(shape)
        mask = DrivableAreaProperty.CONTINUOUS_COLLISION_MASK if line_type != LineType.BROKEN else DrivableAreaProperty.BROKEN_COLLISION_MASK
        body_np.node().setIntoCollideMask(mask)
        self.static_nodes.append(body_np.node())

        body_np.setPos(panda_position(middle, DrivableAreaProperty.LANE_LINE_GHOST_HEIGHT / 2))
        direction_v = lane_end - lane_start
        # theta = -numpy.arctan2(direction_v[1], direction_v[0])
        theta = -math.atan2(direction_v[1], direction_v[0])

        body_np.setQuat(LQuaternionf(math.cos(theta / 2), 0, 0, math.sin(theta / 2)))

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
            node_name = BodyName.White_continuous_line if color == LineColor.GREY else BodyName.Yellow_continuous_line
        else:
            node_name = BodyName.Broken_line

        # add bullet body for it
        if straight_stripe:
            body_np = parent_np.attachNewNode(node_name)
        else:
            body_node = BulletGhostNode(node_name)
            body_node.set_active(False)
            body_node.setKinematic(False)
            body_node.setStatic(True)
            body_np = parent_np.attachNewNode(body_node)
            # its scale will change by setScale
            body_height = DrivableAreaProperty.LANE_LINE_GHOST_HEIGHT
            shape = BulletBoxShape(
                Vec3(
                    length / 2 if line_type != LineType.BROKEN else length, DrivableAreaProperty.LANE_LINE_WIDTH / 2,
                    body_height
                )
            )
            body_np.node().addShape(shape)
            mask = DrivableAreaProperty.CONTINUOUS_COLLISION_MASK if line_type != LineType.BROKEN else DrivableAreaProperty.BROKEN_COLLISION_MASK
            body_np.node().setIntoCollideMask(mask)
            self.static_nodes.append(body_np.node())

        # position and heading
        body_np.setPos(panda_position(middle, DrivableAreaProperty.LANE_LINE_GHOST_HEIGHT / 2))
        direction_v = lane_end - lane_start
        # theta = -numpy.arctan2(direction_v[1], direction_v[0])
        theta = -math.atan2(direction_v[1], direction_v[0])
        body_np.setQuat(LQuaternionf(math.cos(theta / 2), 0, 0, math.sin(theta / 2)))

        if self.render:
            # For visualization
            lane_line = self.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
            lane_line.setScale(length, DrivableAreaProperty.LANE_LINE_WIDTH, DrivableAreaProperty.LANE_LINE_THICKNESS)
            lane_line.setPos(Vec3(0, 0 - DrivableAreaProperty.LANE_LINE_GHOST_HEIGHT / 2))
            lane_line.reparentTo(body_np)
            body_np.set_color(color)

    def _add_sidewalk2bullet(self, lane_start, lane_end, middle, radius=0.0, direction=0):
        length = norm(lane_end[0] - lane_start[0], lane_end[1] - lane_start[1])
        body_node = BulletRigidBodyNode(BodyName.Sidewalk)
        body_node.setKinematic(False)
        body_node.setStatic(True)
        side_np = self.sidewalk_node_path.attachNewNode(body_node)
        shape = BulletBoxShape(Vec3(1 / 2, 1 / 2, 1 / 2))
        body_node.addShape(shape)
        body_node.setIntoCollideMask(self.SIDEWALK_COLLISION_MASK)
        self.dynamic_nodes.append(body_node)

        if radius == 0:
            factor = 1
        else:
            if direction == 1:
                factor = (1 - self.SIDEWALK_LINE_DIST / radius)
            else:
                factor = (1 + self.SIDEWALK_WIDTH / radius) * (1 + self.SIDEWALK_LINE_DIST / radius)
        direction_v = lane_end - lane_start
        vertical_v = Vector((-direction_v[1], direction_v[0])) / norm(*direction_v)
        middle += vertical_v * (self.SIDEWALK_WIDTH / 2 + self.SIDEWALK_LINE_DIST)
        side_np.setPos(panda_position(middle, 0))
        theta = -math.atan2(direction_v[1], direction_v[0])
        side_np.setQuat(LQuaternionf(math.cos(theta / 2), 0, 0, math.sin(theta / 2)))
        side_np.setScale(length * factor, self.SIDEWALK_WIDTH, self.SIDEWALK_THICKNESS * (1 + 0.1 * np.random.rand()))
        if self.render:
            side_np.setTexture(self.ts_color, self.side_texture)
            self.sidewalk.instanceTo(side_np)

    def _add_lane_surface(self, from_: str, to_: str, lanes: List):
        """
        Add the land surface to world, this surface will record the lane information, like index
        :param from_: From node
        :param to_: To Node
        :param lanes: All lanes of this road
        """

        if isinstance(lanes[0], StraightLane):
            for index, lane in enumerate(lanes):
                middle = lane.position(lane.length / 2, 0)
                end = lane.position(lane.length, 0)
                direction_v = end - middle
                theta = -math.atan2(direction_v[1], direction_v[0])
                width = lane.width_at(0) + self.SIDEWALK_LINE_DIST * 2
                length = lane.length
                self._add_lane2bullet(middle, width, length, theta, lane, (from_, to_, index))
        elif isinstance(lanes[0], CircularLane):
            for index, lane in enumerate(lanes):
                segment_num = int(lane.length / self.CIRCULAR_SEGMENT_LENGTH)
                for i in range(segment_num):
                    middle = lane.position(lane.length * (i + .5) / segment_num, 0)
                    end = lane.position(lane.length * (i + 1) / segment_num, 0)
                    direction_v = end - middle
                    theta = -math.atan2(direction_v[1], direction_v[0])
                    width = lane.width_at(0) + self.SIDEWALK_LINE_DIST * 2
                    length = lane.length
                    self._add_lane2bullet(middle, width, length * 1.3 / segment_num, theta, lane, (from_, to_, index))
        elif isinstance(lanes[0], WayPointLane):
            for index, lane in enumerate(lanes):
                for segment in lane.segment_property:
                    lane_start = segment["start_point"]
                    lane_end = segment["end_point"]
                    middle = (lane_start + lane_end) / 2
                    direction_v = lane_end - middle
                    theta = -math.atan2(direction_v[1], direction_v[0])
                    width = lane.width_at(0)
                    length = segment["length"]
                    self._add_lane2bullet(middle, width, length, theta, lane, (from_, to_, index))

    def _add_lane2bullet(self, middle, width, length, theta, lane: Union[StraightLane, CircularLane], lane_index):
        """
        Add lane visualization and body for it
        :param middle: Middle point
        :param width: Lane width
        :param length: Segment length
        :param theta: Rotate theta
        :param lane: Lane info
        :return: None
        """
        length += 0.1
        lane.index = lane_index
        segment_np = NodePath(BaseRigidBodyNode(lane, BodyName.Lane))
        segment_node = segment_np.node()
        segment_node.set_active(False)
        segment_node.setKinematic(False)
        segment_node.setStatic(True)
        shape = BulletBoxShape(Vec3(length / 2, 0.1, width / 2))
        segment_node.addShape(shape)
        self.static_nodes.append(segment_node)
        segment_np.setPos(panda_position(middle, -0.1))
        segment_np.setQuat(
            LQuaternionf(
                math.cos(theta / 2) * math.cos(-math.pi / 4),
                math.cos(theta / 2) * math.sin(-math.pi / 4), -math.sin(theta / 2) * math.cos(-math.pi / 4),
                math.sin(theta / 2) * math.cos(-math.pi / 4)
            )
        )
        segment_np.reparentTo(self.lane_node_path)
        if self.render:
            cm = CardMaker('card')
            cm.setFrame(-length / 2, length / 2, -width / 2, width / 2)
            cm.setHasNormals(True)
            cm.setUvRange((0, 0), (length / 20, width / 10))
            card = self.lane_vis_node_path.attachNewNode(cm.generate())
            card.setPos(panda_position(middle, np.random.rand() * 0.01 - 0.01))

            card.setQuat(
                LQuaternionf(
                    math.cos(theta / 2) * math.cos(-math.pi / 4),
                    math.cos(theta / 2) * math.sin(-math.pi / 4), -math.sin(theta / 2) * math.cos(-math.pi / 4),
                    math.sin(theta / 2) * math.cos(-math.pi / 4)
                )
            )
            card.setTransparency(TransparencyAttrib.MMultisample)
            card.setTexture(self.ts_color, self.road_texture)

    def add_body(self, physics_body):
        raise DeprecationWarning(
            "Different from common objects like vehicle/traffic sign, Block has several bodies!"
            "Therefore, you should create BulletBody and then add them to self.dynamics_nodes "
            "manually. See in construct() method"
        )
