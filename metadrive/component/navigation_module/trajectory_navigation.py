from collections import deque
from metadrive.constants import CamMask

import numpy as np
from panda3d.core import NodePath, Material
from metadrive.engine.logger import get_logger
from metadrive.component.navigation_module.base_navigation import BaseNavigation
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.coordinates_shift import panda_vector
from metadrive.utils.math import norm, clip
from metadrive.utils.math import panda_vector
from metadrive.utils.math import wrap_to_pi


class TrajectoryNavigation(BaseNavigation):
    """
    This module enabling follow a given reference trajectory given a map
    """
    DISCRETE_LEN = 2  # m
    CHECK_POINT_INFO_DIM = 2
    NUM_WAY_POINT = 10
    NAVI_POINT_DIST = 30  # m, used to clip value, should be greater than DISCRETE_LEN * MAX_NUM_WAY_POINT

    def __init__(
        self,
        show_navi_mark: bool = False,
        show_dest_mark=False,
        show_line_to_dest=False,
        panda_color=None,
        name=None,
        vehicle_config=None
    ):
        if show_dest_mark or show_line_to_dest:
            get_logger().warning("show_dest_mark and show_line_to_dest are not supported in TrajectoryNavigation")
        super(TrajectoryNavigation, self).__init__(
            show_navi_mark=show_navi_mark,
            show_dest_mark=show_dest_mark,
            show_line_to_dest=show_line_to_dest,
            panda_color=panda_color,
            name=name,
            vehicle_config=vehicle_config
        )
        if self.origin is not None:
            self.origin.hide(CamMask.RgbCam | CamMask.Shadow | CamMask.DepthCam | CamMask.SemanticCam)

        self._route_completion = 0
        self.checkpoints = None  # All check points

        # for compatibility
        self.next_ref_lanes = None

        # override the show navi mark function here
        self._navi_point_model = None
        self._ckpt_vis_models = None
        if show_navi_mark and self._show_navi_info:
            self._ckpt_vis_models = [NodePath(str(i)) for i in range(self.NUM_WAY_POINT)]
            for model in self._ckpt_vis_models:
                if self._navi_point_model is None:
                    self._navi_point_model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
                    self._navi_point_model.setScale(0.5)
                    # if self.engine.use_render_pipeline:
                    material = Material()
                    material.setBaseColor((19 / 255, 212 / 255, 237 / 255, 1))
                    material.setShininess(16)
                    material.setEmission((0.2, 0.2, 0.2, 0.2))
                    self._navi_point_model.setMaterial(material, True)
                self._navi_point_model.instanceTo(model)
                model.reparentTo(self.origin)

        # should be updated every step after calling update_localization
        self.last_current_long = deque([0.0, 0.0], maxlen=2)
        self.last_current_lat = deque([0.0, 0.0], maxlen=2)
        self.last_current_heading_theta_at_long = deque([0.0, 0.0], maxlen=2)

    def reset(self, vehicle):
        super(TrajectoryNavigation, self).reset(current_lane=self.reference_trajectory)
        self.set_route()

    @property
    def reference_trajectory(self):
        return self.engine.map_manager.current_sdc_route

    @property
    def current_ref_lanes(self):
        return [self.reference_trajectory]

    def set_route(self):
        self.checkpoints = self.discretize_reference_trajectory()
        num_way_point = min(len(self.checkpoints), self.NUM_WAY_POINT)

        self._navi_info.fill(0.0)
        self.next_ref_lanes = None
        if self._dest_node_path is not None:
            check_point = self.reference_trajectory.end
            self._dest_node_path.setPos(panda_vector(check_point[0], check_point[1], 1))

    def discretize_reference_trajectory(self):
        ret = []
        length = self.reference_trajectory.length
        num = int(length / self.DISCRETE_LEN)
        for i in range(num):
            ret.append(self.reference_trajectory.position(i * self.DISCRETE_LEN, 0))
        ret.append(self.reference_trajectory.end)
        return ret

    def update_localization(self, ego_vehicle):
        """
        It is called every step
        """
        if self.reference_trajectory is None:
            return

        # Update ckpt index
        long, lat = self.reference_trajectory.local_coordinates(ego_vehicle.position)
        heading_theta_at_long = self.reference_trajectory.heading_theta_at(long)
        self.last_current_heading_theta_at_long.append(heading_theta_at_long)
        self.last_current_long.append(long)
        self.last_current_lat.append(lat)

        next_idx = max(int(long / self.DISCRETE_LEN) + 1, 0)
        next_idx = min(next_idx, len(self.checkpoints) - 1)
        end_idx = min(next_idx + self.NUM_WAY_POINT, len(self.checkpoints))
        ckpts = self.checkpoints[next_idx:end_idx]
        diff = self.NUM_WAY_POINT - len(ckpts)
        assert diff >= 0, "Number of Navigation points error!"
        if diff > 0:
            ckpts += [self.checkpoints[-1] for _ in range(diff)]

        # target_road_1 is the road segment the vehicle is driving on.
        self._navi_info.fill(0.0)
        for k, ckpt in enumerate(ckpts[1:]):
            start = k * self.CHECK_POINT_INFO_DIM
            end = (k + 1) * self.CHECK_POINT_INFO_DIM
            self._navi_info[start:end], lanes_heading = self._get_info_for_checkpoint(ckpt, ego_vehicle)
            if self._show_navi_info and self._ckpt_vis_models is not None:
                pos_of_goal = ckpt
                self._ckpt_vis_models[k].setPos(panda_vector(pos_of_goal[0], pos_of_goal[1], self.MARK_HEIGHT))
                self._ckpt_vis_models[k].setH(self._goal_node_path.getH() + 3)

        self._navi_info[end] = clip((lat / self.engine.global_config["max_lateral_dist"] + 1) / 2, 0.0, 1.0)
        self._navi_info[end + 1] = clip(
            (wrap_to_pi(heading_theta_at_long - ego_vehicle.heading_theta) / np.pi + 1) / 2, 0.0, 1.0
        )

        # Use RC as the only criterion to determine arrival in Scenario env.
        self._route_completion = long / self.reference_trajectory.length

        if self._show_navi_info:
            # Whether to visualize little boxes in the scene denoting the checkpoints
            pos_of_goal = ckpts[1]
            self._goal_node_path.setPos(panda_vector(pos_of_goal[0], pos_of_goal[1], self.MARK_HEIGHT))
            self._goal_node_path.setH(self._goal_node_path.getH() + 3)
            # self.navi_arrow_dir = [lanes_heading1, lanes_heading2]
            dest_pos = self._dest_node_path.getPos()
            self._draw_line_to_dest(start_position=ego_vehicle.position, end_position=(dest_pos[0], dest_pos[1]))
            navi_pos = self._goal_node_path.getPos()
            self._draw_line_to_navi(start_position=ego_vehicle.position, end_position=(navi_pos[0], navi_pos[1]))

    def get_current_lateral_range(self, current_position, engine) -> float:
        return self.current_lane.width * 2

    def get_current_lane_width(self) -> float:
        return self.current_lane.width

    def get_current_lane_num(self) -> float:
        return 1

    @classmethod
    def _get_info_for_checkpoint(cls, checkpoint, ego_vehicle):
        navi_information = []
        # Project the checkpoint position into the target vehicle's coordination, where
        # +x is the heading and +y is the right hand side.
        dir_vec = checkpoint - ego_vehicle.position  # get the vector from center of vehicle to checkpoint
        dir_norm = norm(dir_vec[0], dir_vec[1])
        if dir_norm > cls.NAVI_POINT_DIST:  # if the checkpoint is too far then crop the direction vector
            dir_vec = dir_vec / dir_norm * cls.NAVI_POINT_DIST
        ckpt_in_heading, ckpt_in_rhs = ego_vehicle.convert_to_local_coordinates(
            dir_vec, 0.0
        )  # project to ego vehicle's coordination

        # Dim 1: the relative position of the checkpoint in the target vehicle's heading direction.
        navi_information.append(clip((ckpt_in_heading / cls.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))

        # Dim 2: the relative position of the checkpoint in the target vehicle's right hand side direction.
        navi_information.append(clip((ckpt_in_rhs / cls.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))

        return navi_information

    def destroy(self):
        self.checkpoints = None
        # self.current_ref_lanes = None
        self.next_ref_lanes = None
        self.final_lane = None
        self._current_lane = None
        # self.reference_trajectory = None
        super(TrajectoryNavigation, self).destroy()

    def before_reset(self):
        self.checkpoints = None
        # self.current_ref_lanes = None
        self.next_ref_lanes = None
        self.final_lane = None
        self._current_lane = None
        # self.reference_trajectory = None

    @property
    def route_completion(self):
        return self._route_completion

    @classmethod
    def get_navigation_info_dim(cls):
        return cls.NUM_WAY_POINT * cls.CHECK_POINT_INFO_DIM + 2

    @property
    def last_longitude(self):
        return self.last_current_long[0]

    @property
    def current_longitude(self):
        return self.last_current_long[1]

    @property
    def last_lateral(self):
        return self.last_current_lat[0]

    @property
    def current_lateral(self):
        return self.last_current_lat[1]

    @property
    def last_heading_theta_at_long(self):
        return self.last_current_heading_theta_at_long[0]

    @property
    def current_heading_theta_at_long(self):
        return self.last_current_heading_theta_at_long[1]
