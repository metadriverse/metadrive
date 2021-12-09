import math
from typing import Dict, List

from metadrive.base_class.base_object import BaseObject
from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.lane.waypoint_lane import WayPointLane
from metadrive.component.road_network import Road
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.constants import BodyName, CamMask, LineType, LineColor, DrivableAreaProperty
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.core.physics_world import PhysicsWorld
from metadrive.utils.coordinates_shift import panda_position
from metadrive.utils.math_utils import norm
from panda3d.bullet import BulletBoxShape, BulletGhostNode
from panda3d.core import Vec3, LQuaternionf, Vec4, TextureStage, RigidBodyCombiner, \
    SamplerState, NodePath


class BaseBlock(BaseObject, DrivableAreaProperty):
    """
    Block is a driving area consisting of several roads
    Note: overriding the _sample() function to fill block_network/respawn_roads in subclass
    Call Block.construct_block() to add it to world
    """

    ID = "B"

    def __init__(
        self, block_index: int, global_network: NodeRoadNetwork, random_seed, ignore_intersection_checking=False
    ):
        super(BaseBlock, self).__init__(str(block_index) + self.ID, random_seed, escape_random_seed_assertion=True)
        # block information
        assert self.ID is not None, "Each Block must has its unique ID When define Block"
        assert len(self.ID) == 1, "Block ID must be a character "

        self.block_index = block_index
        self.ignore_intersection_checking = ignore_intersection_checking

        # each block contains its own road network and a global network
        self._global_network = global_network
        self.block_network = self.block_network_type()

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

        Note: Override the create_in_world() function instead of this one, since this method severing as a wrapper to
        help improve efficiency
        """
        self.lane_line_node_path = NodePath(RigidBodyCombiner(self.name + "_lane_line"))
        self.sidewalk_node_path = NodePath(RigidBodyCombiner(self.name + "_sidewalk"))
        self.lane_node_path = NodePath(RigidBodyCombiner(self.name + "_lane"))
        self.lane_vis_node_path = NodePath(RigidBodyCombiner(self.name + "_lane_vis"))
        self.create_in_world()
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

    def create_in_world(self):
        """
        Create lane in the panda3D world
        """
        raise NotImplementedError

    def add_body(self, physics_body):
        raise DeprecationWarning(
            "Different from common objects like vehicle/traffic sign, Block has several bodies!"
            "Therefore, you should create BulletBody and then add them to self.dynamics_nodes "
            "manually. See in construct() method"
        )

    def get_state(self) -> Dict:
        """
        The record of Block type is not same as other objects
        """
        return {}

    def set_state(self, state: Dict):
        """
        Block type can not set state currently
        """
        pass

    def _add_lane_line(self, lane: AbstractLane, colors: List[Vec4], contruct_two_side=True):
        raise DeprecationWarning("Leave for argoverse using")
        if isinstance(lane, WayPointLane):
            parent_np = self.lane_line_node_path
            lane_width = lane.width_at(0)
            for c, i in enumerate([-1, 1]):
                line_color = colors[c]
                acc_length = 0
                if lane.line_types[c] == LineType.CONTINUOUS:
                    for segment in lane.segment_property:
                        lane_start = lane.position(acc_length, i * lane_width / 2)
                        acc_length += segment["length"]
                        lane_end = lane.position(acc_length, i * lane_width / 2)
                        middle = (lane_start + lane_end) / 2
                        self._add_lane_line2bullet(
                            lane_start, lane_end, middle, parent_np, line_color, lane.line_types[c]
                        )

    def _add_box_body(self, lane_start, lane_end, middle, parent_np: NodePath, line_type, line_color):
        raise DeprecationWarning("Useless, currently")
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

    @property
    def block_network_type(self):
        """
        There are two type of road network to describe the relation of all lanes, override this func to assign one when
        you are building your own block.
        return: roadnetwork
        """
        raise NotImplementedError

    def destroy(self):
        if self.block_network is not None:
            self.block_network.destroy()
            self.block_network = None
        super(BaseBlock, self).destroy()
