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
                self.control_object.set_throttle_brake(float(info["throttle_brake"]))
        if "steering" in info:
            if hasattr(self.control_object, "set_steering"):
                self.control_object.set_steering(float(info["steering"]))
        self.control_object.set_position(info["position"])
        self.control_object.set_velocity(info["velocity"], in_local_frame=self._velocity_local_frame)
        self.control_object.set_heading_theta(info["heading"])
        self.control_object.set_angular_velocity(info["angular_velocity"])

        return None  # Return None action so the base vehicle will not overwrite the steering & throttle


WaymoReplayTrafficParticipantPolicy = ReplayTrafficParticipantPolicy
ScenarioReplayTrafficParticipantPolicy = ReplayTrafficParticipantPolicy


class ReplayEgoCarPolicy(ReplayTrafficParticipantPolicy):
    def get_trajectory_info(self, trajectory):
        # Directly get trajectory from data manager
        trajectory_data = self.engine.data_manager.current_scenario["tracks"]
        sdc_track_index = str(self.engine.data_manager.current_scenario["metadata"]["sdc_id"])
        # if self.engine.data_manager.current_scenario["metadata"]["dataset"] == "nuplan":
        #     # nuplan local frame velocity
        #     self._velocity_local_frame = True
        ret = []
        for i in range(len(trajectory_data[sdc_track_index]["state"]["position"])):
            ret.append(parse_object_state(
                trajectory_data[sdc_track_index],
                i,
            ))
        return ret


WaymoReplayEgoCarPolicy = ReplayEgoCarPolicy
ScenarioReplayEgoCarPolicy = ReplayEgoCarPolicy


class NuPlanReplayEgoCarPolicy(ReplayEgoCarPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sim_time_interval = self.engine.data_manager.time_interval
        if not self.control_object.config["no_wheel_friction"]:
            logger.warning("\nNOTE:set no_wheel_friction in vehicle config can make the replay more smooth! \n")

    def get_trajectory_info(self, *args, **kwargs):
        from metadrive.utils.nuplan.parse_object_state import parse_ego_vehicle_state_trajectory
        scenario = self.engine.data_manager.current_scenario
        return parse_ego_vehicle_state_trajectory(scenario, self.engine.current_map.nuplan_center)

    def act(self, *args, **kwargs):
        if self.episode_step >= len(self.traj_info):
            return

        self.control_object.set_position(self.traj_info[int(self.episode_step)]["position"])
        if self.episode_step < len(self.traj_info) - 1:
            velocity = self.traj_info[int(self.episode_step + 1)]["position"] - self.traj_info[int(self.episode_step
                                                                                                   )]["position"]
            velocity /= self.sim_time_interval
            self.control_object.set_velocity(velocity, in_local_frame=False)
        else:
            velocity = self.traj_info[int(self.episode_step)]["velocity"]
            self.control_object.set_velocity(velocity, in_local_frame=True)
        # self.control_object.set_velocity(self.traj_info[int(self.episode_step)]["velocity"])

        this_heading = self.traj_info[int(self.episode_step)]["heading"]
        angular_v = self.traj_info[int(self.episode_step)]["angular_velocity"]
        self.control_object.set_heading_theta(this_heading)
        self.control_object.set_angular_velocity(angular_v)

        return [0, 0]


class NuPlanReplayTrafficParticipantPolicy(BasePolicy):
    """
    This policy should be used with TrafficParticipantManager Together
    """
    def __init__(self, control_object, fix_height=None, random_seed=None, config=None):
        super(NuPlanReplayTrafficParticipantPolicy, self).__init__(control_object, random_seed, config)
        self.fix_height = fix_height

    def act(self, obj_state, *args, **kwargs):
        self.control_object.set_position(obj_state["position"], self.fix_height)
        self.control_object.set_heading_theta(obj_state["heading"])
        self.control_object.set_velocity(obj_state["velocity"])
        return [0, 0]
