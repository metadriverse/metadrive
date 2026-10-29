import gymnasium as gym
from metadrive.engine.logger import get_logger
from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.engine.engine_utils import get_global_config
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.utils.math import wrap_to_pi

logger = get_logger()


class LaneChangePolicy(EnvInputPolicy):
    def __init__(self, obj, seed):
        # Since control object may change
        super(LaneChangePolicy, self).__init__(obj, seed)
        self.discrete_action = self.engine.global_config["discrete_action"]
        assert self.discrete_action, "Must set discrete_action=True for using this control policy"
        self.use_multi_discrete = self.engine.global_config["use_multi_discrete"]
        self.steering_unit = 1.0
        self.throttle_unit = 2.0 / (
            self.engine.global_config["discrete_throttle_dim"] - 1
        )  # for discrete actions space
        self.discrete_steering_dim = 3  # only left or right
        self.discrete_throttle_dim = self.engine.global_config["discrete_throttle_dim"]
        logger.info("The discrete_steering_dim for {} is set to 3 [left, keep, right]".format(self.name))

        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.3, .002, 0.05)

    def act(self, agent_id):
        action = super(LaneChangePolicy, self).act(agent_id)
        steering = action[0]
        throttle = action[1]
        current_lane = self.control_object.navigation.current_lane
        if steering == 0.0:
            target_lane = current_lane
        elif steering == 1.0:
            target_lane = self.control_object.navigation.current_ref_lanes[max(current_lane.index[-1] - 1, 0)]
        elif steering == -1.0:
            lane_num = len(self.control_object.navigation.current_ref_lanes)
            target_lane = self.control_object.navigation.current_ref_lanes[
                min(current_lane.index[-1] + 1, lane_num - 1)]
        else:
            raise ValueError("Steering Error, can only be in [-1, 0, 1]")
        action = [self.steering_control(target_lane), throttle]
        self.action_info["action"] = action
        return action

    @classmethod
    def get_input_space(cls):
        """
       The Input space is a class attribute
       """
        engine_global_config = get_global_config()
        assert engine_global_config["discrete_action"]
        discrete_throttle_dim = engine_global_config["discrete_throttle_dim"]
        use_multi_discrete = engine_global_config["use_multi_discrete"]
        discrete_steering_dim = 3

        if use_multi_discrete:
            return gym.spaces.MultiDiscrete([discrete_steering_dim, discrete_throttle_dim])
        else:
            return gym.spaces.Discrete(discrete_steering_dim * discrete_throttle_dim)

    def steering_control(self, target_lane) -> float:
        # heading control following a lateral distance control
        ego_vehicle = self.control_object
        long, lat = target_lane.local_coordinates(ego_vehicle.position)
        lane_heading = target_lane.heading_theta_at(long + 1)
        v_heading = ego_vehicle.heading_theta
        steering = self.heading_pid.get_result(-wrap_to_pi(lane_heading - v_heading))
        steering += self.lateral_pid.get_result(-lat)
        return float(steering)
