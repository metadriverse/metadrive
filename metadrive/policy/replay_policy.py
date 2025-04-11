import logging

from metadrive.policy.base_policy import BasePolicy
from metadrive.scenario.parse_object_state import parse_object_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReplayTrafficParticipantPolicy(BasePolicy):
    """
       Replay policy from Real data. For adding new policy, overwrite get_trajectory_info()
       This policy is designed for Waymo Policy by default
       """
    DEBUG_MARK_COLOR = (3, 140, 252, 255)

    def __init__(self, control_object, track, random_seed=None):
        super(ReplayTrafficParticipantPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.start_index = 0
        self._velocity_local_frame = False
        self.traj_info = self.get_trajectory_info(track)

    @property
    def is_current_step_valid(self):
        return self.traj_info[self.episode_step] is not None

    def get_trajectory_info(self, track):
        ret = []
        for i in range(self.engine.data_manager.current_scenario_length):
            # a trick for saving computation
            if i < self.episode_step:
                ret.append(None)
            else:
                state = parse_object_state(track, i)
                if not state["valid"]:
                    ret.append(None)
                else:
                    ret.append(state)
        return ret

    def act(self, *args, **kwargs):
        index = max(int(self.episode_step), 0)
        if index >= len(self.traj_info):
            return None

        info = self.traj_info[index]

        # Before step
        # Warning by LQY: Don't call before step here! Before step should be called by manager
        # action = self.traj_info[int(self.episode_step)].get("action", None)
        # self.control_object.before_step(action)

        if not bool(info["valid"]):
            return None  # Return None action so the base vehicle will not overwrite the steering & throttle

        if "throttle_brake" in info:
            if hasattr(self.control_object, "set_throttle_brake"):
                self.control_object.set_throttle_brake(float(info["throttle_brake"].item()))
        if "steering" in info:
            if hasattr(self.control_object, "set_steering"):
                self.control_object.set_steering(float(info["steering"].item()))
        self.control_object.set_position(info["position"])
        self.control_object.set_velocity(info["velocity"], in_local_frame=self._velocity_local_frame)
        self.control_object.set_heading_theta(info["heading"])
        self.control_object.set_angular_velocity(info["angular_velocity"])

        # If set_static, then the agent will not "fall from the sky".
        # However, the physics engine will not update the position of the agent.
        # So in the visualization, the image will be very chunky as the agent will not suddenly move to the next
        # position for each step.

        if self.engine.global_config.get("set_static", False):
            self.control_object.set_static(True)

        return None  # Return None action so the base vehicle will not overwrite the steering & throttle


class ReplayEgoCarPolicy(ReplayTrafficParticipantPolicy):
    def get_trajectory_info(self, trajectory):
        # Directly get trajectory from data manager
        trajectory_data = self.engine.data_manager.current_scenario["tracks"]
        sdc_track_index = str(self.engine.data_manager.current_scenario["metadata"]["sdc_id"])
        ret = []
        for i in range(len(trajectory_data[sdc_track_index]["state"]["position"])):
            ret.append(parse_object_state(
                trajectory_data[sdc_track_index],
                i,
            ))
        return ret


import copy
import numpy as np
from metadrive.utils.waypoint_utils import interpolate_velocities, interpolate_angular_velocities, interpolate_headings


class WayPointPolicy(ReplayEgoCarPolicy):
    """
    This policy will have the trajectory data being overwritten on the fly.
    """
    def __init__(self, control_object, track, random_seed=None):
        super(WayPointPolicy, self).__init__(control_object=control_object, track=track, random_seed=random_seed)
        self.online_traj_info = copy.deepcopy(self.traj_info)

    def act(self, *args, **kwargs):
        assert "agent_id" in kwargs.keys() and "actions" in kwargs.keys()
        index = max(int(self.episode_step), 0)
        if index >= len(self.traj_info):
            return None
        """
        Example step
        self.waypoint_info is gonna be a list composed of values in the following format
        {
            'angular_velocity': 0.025234256098722874, 
            'heading': -1.9234410099058392, 
            'heading_theta': -1.9234410099058392, 
            'height': 1.56, 
            'length': 1.73, 
            'position': [0. 0.], 
            'valid': 1.0, 
            'vehicle_class': None, 
            'velocity': [-3.16746009 -8.62107663], 
            'width': 4.08
        }
        """
        # start overwriting the traj_info
        # actions will be a list of dicts also
        if kwargs["actions"] is not None and kwargs["actions"]["default_agent"] is not None:
            # User input is in ego's coordinate, convert it to world coordinate
            ego_positions = kwargs["actions"]["default_agent"]["position"].tolist()
            world_positions = np.array([self.control_object.convert_to_world_coordinates(point, self.control_object.position) for point in ego_positions])
            # world headings, velocities, and angular_velocities from world waypoints
            headings = np.array(interpolate_headings(world_positions))
            angular_velocities = np.array(interpolate_angular_velocities(headings, 0.1))
            velocities = np.array(interpolate_velocities(world_positions, 0.1))
            duration = len(kwargs["actions"]["default_agent"]["position"])
            for i in range(duration):
                # Overwrite the online buffer. It will not be longer than the original trajectory.
                if index + i >= len(self.online_traj_info):
                    continue
                self.online_traj_info[index + i]["angular_velocity"]   = angular_velocities[i]
                self.online_traj_info[index + i]["heading"]            = headings[i]
                self.online_traj_info[index + i]["heading_theta"]      = headings[i]
                self.online_traj_info[index + i]["position"]           = world_positions[i]
                self.online_traj_info[index + i]["velocity"]           = velocities[i]
            # For the rewrite the tailing information
            for i in range(index + duration, len(self.online_traj_info)):
                self.online_traj_info[i]["angular_velocity"]   = angular_velocities[-1]
                self.online_traj_info[i]["heading"]            = headings[-1]
                self.online_traj_info[i]["heading_theta"]      = headings[-1]
                self.online_traj_info[i]["position"]           = world_positions[-1]
                self.online_traj_info[i]["velocity"]           = velocities[-1]
        info = self.online_traj_info[index]

        # Before step
        # Warning by LQY: Don't call before step here! Before step should be called by manager
        # action = self.traj_info[int(self.episode_step)].get("action", None)
        # self.control_object.before_step(action)

        if not bool(info["valid"]):
            return None  # Return None action so the base vehicle will not overwrite the steering & throttle

        if "throttle_brake" in info:
            if hasattr(self.control_object, "set_throttle_brake"):
                self.control_object.set_throttle_brake(float(info["throttle_brake"].item()))
        if "steering" in info:
            if hasattr(self.control_object, "set_steering"):
                self.control_object.set_steering(float(info["steering"].item()))

        self.control_object.set_position(info["position"])
        self.control_object.set_velocity(info["velocity"], in_local_frame=self._velocity_local_frame)
        self.control_object.set_heading_theta(info["heading"])
        self.control_object.set_angular_velocity(info["angular_velocity"])

        # If set_static, then the agent will not "fall from the sky".
        # However, the physics engine will not update the position of the agent.
        # So in the visualization, the image will be very chunky as the agent will not suddenly move to the next
        # position for each step.

        if self.engine.global_config.get("set_static", False):
            self.control_object.set_static(True)

        return None  # Return None action so the base vehicle will not overwrite the steering & throttle
