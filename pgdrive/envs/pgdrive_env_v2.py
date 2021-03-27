import logging
import os.path as osp

import numpy as np
from pgdrive.constants import DEFAULT_AGENT
from pgdrive.envs.pgdrive_env import PGDriveEnv as PGDriveEnvV1
from pgdrive.scene_manager.traffic_manager import TrafficMode
from pgdrive.utils import PGConfig, clip

pregenerated_map_file = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "assets", "maps", "PGDrive-maps.json")


class PGDriveEnvV2(PGDriveEnvV1):
    DEFAULT_AGENT = DEFAULT_AGENT

    @staticmethod
    def default_config() -> PGConfig:
        config = PGDriveEnvV1.default_config()
        config.update(
            dict(
                # ===== Traffic =====
                traffic_density=0.1,
                traffic_mode=TrafficMode.Trigger,  # "reborn", "trigger", "hybrid"
                random_traffic=False,  # Traffic is randomized at default.

                # ===== Cost Scheme =====
                crash_vehicle_cost=1.,
                crash_object_cost=1.,
                out_of_road_cost=1.,

                # ===== Reward Scheme =====
                # See: https://github.com/decisionforce/pgdrive/issues/283
                success_reward=10.0,
                out_of_road_penalty=5.0,
                crash_vehicle_penalty=5.0,
                crash_object_penalty=5.0,
                acceleration_penalty=0.0,
                driving_reward=1.0,
                general_penalty=0.0,
                speed_reward=0.5,
                use_lateral=False,

                # See: https://github.com/decisionforce/pgdrive/issues/297
                vehicle_config=dict(lidar=dict(num_lasers=120, distance=50, num_others=0)),

                # Disable map loading!
                load_map_from_json=False,
                _load_map_from_json="",
            ),
            allow_overwrite=True
        )
        return config

    def __init__(self, config: dict = None):
        super(PGDriveEnvV2, self).__init__(config=config)

    def done_function(self, vehicle):
        done = False
        done_info = dict(crash_vehicle=False, crash_object=False, out_of_road=False, arrive_dest=False)
        if vehicle.arrive_destination:
            done = True
            logging.info("Episode ended! Reason: arrive_dest.")
            done_info["arrive_dest"] = True
        elif vehicle.crash_vehicle:
            done = True
            logging.info("Episode ended! Reason: crash. ")
            done_info["crash_vehicle"] = True
        elif vehicle.out_of_route or not vehicle.on_lane or vehicle.crash_sidewalk:
            done = True
            logging.info("Episode ended! Reason: out_of_road.")
            done_info["out_of_road"] = True
        elif vehicle.crash_object:
            done = True
            done_info["crash_object"] = True

        # for compatibility
        # crash almost equals to crashing with vehicles
        done_info["crash"] = done_info["crash_vehicle"] or done_info["crash_object"]
        return done, done_info

    def cost_function(self, vehicle):
        step_info = dict()
        step_info["cost"] = 0
        if vehicle.crash_vehicle:
            step_info["cost"] = self.config["crash_vehicle_cost"]
        elif vehicle.crash_object:
            step_info["cost"] = self.config["crash_object_cost"]
        elif vehicle.out_of_route:
            step_info["cost"] = self.config["out_of_road_cost"]
        return step_info['cost'], step_info

    def reward_function(self, vehicle):
        """
        Override this func to get a new reward function
        :param vehicle: BaseVehicle
        :return: reward
        """
        step_info = dict()

        # Reward for moving forward in current lane
        current_lane = vehicle.lane
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        reward = 0.0

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral"]:
            lateral_factor = clip(
                1 - 2 * abs(lateral_now) / vehicle.routing_localization.get_current_lane_width(), 0.0, 1.0
            )
        else:
            lateral_factor = 1.0

        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor

        reward += self.config["speed_reward"] * (vehicle.speed / vehicle.max_speed)
        step_info["step_reward"] = reward

        if vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        elif vehicle.out_of_route:
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.arrive_destination:
            reward = +self.config["success_reward"]
        return reward, step_info

    def extra_step_info(self, step_infos):
        return step_infos

    def _get_reset_return(self):
        ret = {}
        self.for_each_vehicle(lambda v: v.update_state())
        for v_id, v in self.vehicles.items():
            self.observations[v_id].reset(self, v)
            ret[v_id] = self.observations[v_id].observe(v)
        return ret[DEFAULT_AGENT] if self.num_agents == 1 else ret


if __name__ == '__main__':

    def _act(env, action):
        assert env.action_space.contains(action)
        obs, reward, done, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isscalar(reward)
        assert isinstance(info, dict)

    env = PGDriveEnvV2({"vehicle_config": {"use_lateral_factor": "Haha", "use_reward_v1": "Fuck"}})
    try:
        obs = env.reset()
        assert env.observation_space.contains(obs)
        _act(env, env.action_space.sample())
        for x in [-1, 0, 1]:
            env.reset()
            for y in [-1, 0, 1]:
                _act(env, [x, y])
    finally:
        env.close()
