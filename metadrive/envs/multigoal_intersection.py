"""
This file implement an intersection environment with multiple goals.
"""
from metadrive.constants import DEFAULT_AGENT
from metadrive.engine.logger import get_logger
from metadrive.manager.base_manager import BaseAgentManager
from metadrive.policy.AI_protect_policy import AIProtectPolicy
from metadrive.policy.idm_policy import TrajectoryIDMPolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy, TakeoverPolicy, TakeoverPolicyWithoutBrake
from metadrive.policy.replay_policy import ReplayTrafficParticipantPolicy
from metadrive.component.pgblock.intersection import InterSection
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.road_network import Road

logger = get_logger()



from metadrive import MetaDriveEnv

from metadrive.utils import clip


GOALS = [
Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3),
-Road(InterSection.node(1, 0, 0), InterSection.node(1, 0, 1)),
-Road(InterSection.node(1, 1, 0), InterSection.node(1, 1, 1)),
-Road(InterSection.node(1, 2, 0), InterSection.node(1, 2, 1)),
]


class MultiGoalIntersectionEnv(MetaDriveEnv):
    @classmethod
    def default_config(cls):
        config = MetaDriveEnv.default_config()
        config.update(
            {
                "start_seed": 0,
                "num_scenarios": 1,
                "map": "X",
            }
        )
        return config

    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        vehicle = self.agents[vehicle_id]
        step_info = dict()

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral_reward"]:
            lateral_factor = clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0

        reward = 0.0
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor * positive_road
        reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) * positive_road

        step_info["step_reward"] = reward
        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        step_info["route_completion"] = vehicle.navigation.route_completion
        return reward, step_info


if __name__ == "__main__":
    config = dict(
        use_render=True,
        manual_control=True,
        vehicle_config=dict(show_lidar=False, show_navi_mark=True, show_line_to_navi_mark=True),
    )
    env = MultiGoalIntersectionEnv(config)
    try:
        o, _ = env.reset()
        for i in range(1, 1000000000):
            o, r, tm, tc, info = env.step([0, 0])
            env.render()
            env.render(mode="topdown")
            if (tm or tc) and info["arrive_dest"]:
                env.reset()
                # env.current_track_agent.expert_takeover = True
    finally:
        env.close()
