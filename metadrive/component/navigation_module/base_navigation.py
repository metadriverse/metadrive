import logging

import numpy as np
from panda3d.core import TransparencyAttrib, LineSegs, NodePath

from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.constants import RENDER_MODE_ONSCREEN, CamMask
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils import get_np_random
from metadrive.utils.coordinates_shift import panda_vector


class BaseNavigation:
    """
    Implement all NotImplemented method for customizing a new navigation module.
    This module interact with the map for finding lanes or expected positions
    """
    NUM_WAY_POINT = 2
    CHECK_POINT_INFO_DIM = 5
    NAVI_POINT_DIST = 50
    PRE_NOTIFY_DIST = 40
    MIN_ALPHA = 0.15
    CKPT_UPDATE_RANGE = 5
    FORCE_CALCULATE = False
    LINE_TO_DEST_HEIGHT = 0.6
    MARK_HEIGHT = 1.2

    def __init__(
        self,
        show_navi_mark: bool = False,
        show_dest_mark=False,
        show_line_to_dest=False,
        panda_color=None,
        name=None,
        vehicle_config=None
    ):
        """
        This class define a helper for localizing vehicles and retrieving navigation information.
        It now only support from first block start to the end node, but can be extended easily.
        """
        self.name = name

        # Make sure these variables are filled when making new subclass
        # self.checkpoints = None
        # self.current_ref_lanes = None
        # self.next_ref_lanes = None
        # self.final_lane = None
        # self.current_lane = None

        self.vehicle_config = vehicle_config

        self._target_checkpoints_index = None
        self._navi_info = np.zeros((self.get_navigation_info_dim(), ), dtype=np.float32)  # navi information res

        # Vis TODO make it beautiful!
        self._show_navi_info = (
            self.engine.mode == RENDER_MODE_ONSCREEN and not self.engine.global_config["debug_physics_world"]
        )
        if self._show_navi_info:
            self.origin = NodePath("navigation_sign")
            self.origin.clearShader()
            self.origin.setShaderAuto()
        else:
            self.origin = None
        if panda_color is not None:
            assert len(panda_color) == 3 and 0 <= panda_color[0] <= 1
            self.navi_mark_color = tuple(panda_color)
        self.navi_arrow_dir = [0, 0]
        self._dest_node_path = None
        self._goal_node_path = None
        self._goal_node_path2 = None

        self._node_path_list = []

        self._line_to_dest = None
        self._line_to_navi = None
        self._show_line_to_dest = show_line_to_dest
        if self._show_navi_info:
            # nodepath
            self._line_to_dest = self.origin.attachNewNode("line")
            self._goal_node_path = self.origin.attachNewNode("target")
            self._goal_node_path2 = self.origin.attachNewNode("target2")
            self._dest_node_path = self.origin.attachNewNode("dest")

            self._node_path_list.append(self._line_to_dest)
            self._node_path_list.append(self._goal_node_path)
            self._node_path_list.append(self._goal_node_path2)
            self._node_path_list.append(self._dest_node_path)

            if show_navi_mark:
                navi_point_model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
                navi_point_model.reparentTo(self._goal_node_path)

                navi_point_model2 = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
                navi_point_model2.reparentTo(self._goal_node_path2)
            if show_dest_mark:
                dest_point_model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
                dest_point_model.reparentTo(self._dest_node_path)
            if show_line_to_dest:
                line_seg = LineSegs("line_to_dest")
                line_seg.setColor(self.navi_mark_color[0], self.navi_mark_color[1], self.navi_mark_color[2], 1.0)
                line_seg.setThickness(4)
                self._dynamic_line_np = NodePath(line_seg.create(True))

                self._node_path_list.append(self._dynamic_line_np)

                self._dynamic_line_np.reparentTo(self.origin)
                self._line_to_dest = line_seg

            show_line_to_navi_mark = self.vehicle_config["show_line_to_navi_mark"]
            self._show_line_to_navi_mark = show_line_to_navi_mark
            if show_line_to_navi_mark:
                line_seg = LineSegs("line_to_dest")
                line_seg.setColor(self.navi_mark_color[0], self.navi_mark_color[1], self.navi_mark_color[2], 1.0)
                line_seg.setThickness(4)
                self._dynamic_line_np_2 = NodePath(line_seg.create(True))
                self._node_path_list.append(self._dynamic_line_np_2)
                self._dynamic_line_np_2.reparentTo(self.origin)
                self._line_to_navi = line_seg

            self._goal_node_path.setTransparency(TransparencyAttrib.M_alpha)
            self._goal_node_path2.setTransparency(TransparencyAttrib.M_alpha)
            self._dest_node_path.setTransparency(TransparencyAttrib.M_alpha)

            self._goal_node_path.setColor(
                self.navi_mark_color[0], self.navi_mark_color[1], self.navi_mark_color[2], 0.7
            )
            self._goal_node_path2.setColor(
                self.navi_mark_color[0], self.navi_mark_color[1], self.navi_mark_color[2], 0.5
            )
            self._dest_node_path.setColor(
                self.navi_mark_color[0], self.navi_mark_color[1], self.navi_mark_color[2], 0.7
            )
            self.origin.hide(CamMask.AllOn)
            # self.origin.hide(CamMask.AllOn)
            # self.origin.show(CamMask.MainCam)
            self.origin.show(CamMask.MainCam)
        logging.debug("Load Vehicle Module: {}".format(self.__class__.__name__))

    def reset(self, current_lane, vehicle_config=None):
        self._current_lane = current_lane
        if vehicle_config is not None:
            self.vehicle_config = vehicle_config

    @property
    def map(self):
        return self.engine.current_map

    @property
    def current_lane(self):
        return self._current_lane

    def get_checkpoints(self):
        """Return next checkpoint and the next next checkpoint"""
        later_middle = (float(self.get_current_lane_num()) / 2 - 0.5) * self.get_current_lane_width()
        ref_lane1 = self.current_ref_lanes[0]
        checkpoint1 = ref_lane1.position(ref_lane1.length, later_middle)
        ref_lane2 = self.next_ref_lanes[0] if self.next_ref_lanes is not None else self.current_ref_lanes[0]
        checkpoint2 = ref_lane2.position(ref_lane2.length, later_middle)
        return checkpoint1, checkpoint2

    def set_route(self, current_lane_index: str, destination: str):
        """
        Find a shortest path from start road to end road
        :param current_lane_index: start road node
        :param destination: end road node or end lane index
        :return: None
        """
        raise NotImplementedError

    def update_localization(self, ego_vehicle):
        """
        It is called every step. This is the core function of navigation module
        """
        raise NotImplementedError

    def get_navi_info(self):
        return self._navi_info

    def destroy(self):
        if self._show_navi_info:
            try:
                if self._line_to_dest is not None:
                    self._line_to_dest.removeNode()
                if self._line_to_navi is not None:
                    self._line_to_navi.removeNode()
            except AttributeError:
                pass
            self._dest_node_path.removeNode()
            self._goal_node_path.removeNode()
            self._goal_node_path2.removeNode()

        for np in self._node_path_list:
            np.detachNode()
            np.removeNode()
        # self.next_ref_lanes = None
        # self.current_ref_lanes = None

    def set_force_calculate_lane_index(self, force: bool):
        self.FORCE_CALCULATE = force

    def __del__(self):
        logging.debug("{} is destroyed".format(self.__class__.__name__))

    def get_current_lateral_range(self, current_position, engine) -> float:
        raise NotImplementedError

    def get_current_lane_width(self) -> float:
        return self.current_lane.width

    def get_current_lane_num(self) -> float:
        return len(self.current_ref_lanes)

    def _ray_lateral_range(self, engine, start_position, dir, length=50):
        """
        It is used to measure the lateral range of special blocks
        :param start_position: start_point
        :param dir: ray direction
        :param length: length of ray
        :return: lateral range [m]
        """
        end_position = start_position[0] + dir[0] * length, start_position[1] + dir[1] * length
        start_position = panda_vector(start_position, z=0.2)
        end_position = panda_vector(end_position, z=0.2)
        mask = FirstPGBlock.CONTINUOUS_COLLISION_MASK
        res = engine.physics_world.static_world.rayTestClosest(start_position, end_position, mask=mask)
        if not res.hasHit():
            return length
        else:
            return res.getHitFraction() * length

    def _draw_line_to_dest(self, start_position, end_position):
        if not self._show_line_to_dest:
            return
        line_seg = self._line_to_dest
        line_seg.moveTo(panda_vector(start_position, self.LINE_TO_DEST_HEIGHT))
        line_seg.drawTo(panda_vector(end_position, self.LINE_TO_DEST_HEIGHT))
        self._dynamic_line_np.removeNode()
        self._dynamic_line_np = NodePath(line_seg.create(False))

        self._node_path_list.append(self._dynamic_line_np)

        self._dynamic_line_np.hide(CamMask.Shadow | CamMask.RgbCam)
        self._dynamic_line_np.reparentTo(self.origin)

    def _draw_line_to_navi(self, start_position, end_position, next_checkpoint=None):
        if not self._show_line_to_navi_mark:
            return
        line_seg = self._line_to_navi
        line_seg.moveTo(panda_vector(start_position, self.LINE_TO_DEST_HEIGHT))
        line_seg.drawTo(panda_vector(end_position, self.LINE_TO_DEST_HEIGHT))

        if next_checkpoint is not None:
            line_seg.drawTo(panda_vector(next_checkpoint, self.LINE_TO_DEST_HEIGHT))

        self._dynamic_line_np_2.removeNode()
        self._dynamic_line_np_2 = NodePath(line_seg.create(False))

        self._node_path_list.append(self._dynamic_line_np_2)

        self._dynamic_line_np_2.hide(CamMask.Shadow | CamMask.RgbCam)
        self._dynamic_line_np_2.reparentTo(self.origin)

    def detach_from_world(self):
        if isinstance(self.origin, NodePath):
            self.origin.detachNode()

    def attach_to_world(self, engine):
        if isinstance(self.origin, NodePath):
            self.origin.reparentTo(engine.render)

    @property
    def engine(self):
        from metadrive.engine.engine_utils import get_engine
        return get_engine()

    def get_state(self):
        return {}

    @classmethod
    def get_navigation_info_dim(cls):
        return cls.NUM_WAY_POINT * cls.CHECK_POINT_INFO_DIM
