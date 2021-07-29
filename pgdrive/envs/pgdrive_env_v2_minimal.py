import math

import gym
import numpy as np

from pgdrive.envs.pgdrive_env_v2 import PGDriveEnvV2
from pgdrive.obs.observation_base import ObservationBase
from pgdrive.obs.state_obs import LidarStateObservation
from pgdrive.utils import Config
from pgdrive.utils.engine_utils import get_engine
from pgdrive.utils.math_utils import norm, clip

DISTANCE = 50


class MinimalObservation(LidarStateObservation):
    _traffic_vehicle_state_dim = 18
    _traffic_vehicle_state_dim_wo_extra = 6

    def __init__(self, config, env):
        super(MinimalObservation, self).__init__(vehicle_config=config)
        self._state_obs_dim = list(self.state_obs.observation_space.shape)[0]
        self._env = env

    @property
    def observation_space(self):
        shape = list(self.state_obs.observation_space.shape)
        shape[0] += 5
        assert self.config["lidar"]["num_others"] >= 0
        if self.config["use_extra_state"]:
            shape[0] = shape[0] + self.config["lidar"]["num_others"] * self._traffic_vehicle_state_dim
        else:
            shape[0] = shape[0] + self.config["lidar"]["num_others"] * self._traffic_vehicle_state_dim_wo_extra
        return gym.spaces.Box(-0.0, 1.0, shape=tuple(shape), dtype=np.float32)

    def observe_ego_state(self, vehicle):
        navi_info = vehicle.routing_localization.get_navi_info()
        ego_state = self.vehicle_state(vehicle)
        return np.concatenate([ego_state, navi_info])

    def vehicle_state(self, vehicle):
        # update out of road
        info = []
        lateral_to_left, lateral_to_right, = vehicle.dist_to_left_side, vehicle.dist_to_right_side
        total_width = float(
            (vehicle.routing_localization.get_current_lane_num() + 1) *
            vehicle.routing_localization.get_current_lane_width()
        )
        lateral_to_left /= total_width
        lateral_to_right /= total_width
        info += [clip(lateral_to_left, 0.0, 1.0), clip(lateral_to_right, 0.0, 1.0)]

        current_reference_lane = vehicle.routing_localization.current_ref_lanes[-1]
        info += [
            vehicle.heading_diff(current_reference_lane),
            # Note: speed can be negative denoting free fall. This happen when emergency brake.
            clip((vehicle.speed + 1) / (vehicle.max_speed + 1), 0.0, 1.0),
            clip((vehicle.steering / vehicle.max_steering + 1) / 2, 0.0, 1.0),
            clip((vehicle.last_current_action[0][0] + 1) / 2, 0.0, 1.0),
            clip((vehicle.last_current_action[0][1] + 1) / 2, 0.0, 1.0)
        ]
        heading_dir_last = vehicle.last_heading_dir
        heading_dir_now = vehicle.heading

        # Add more information about the road
        if hasattr(current_reference_lane, "heading"):
            info.append(clip(current_reference_lane.heading, 0.0, 1.0))
        else:
            info.append(0.0)
        info.append(clip(current_reference_lane.length / DISTANCE, 0.0, 1.0))
        if hasattr(current_reference_lane, "direction"):
            dir = current_reference_lane.direction
            if isinstance(dir, int):
                info.append(self._to_zero_and_one(dir))
                info.append(0.0)
            elif len(dir) == 2:
                info.append(self._to_zero_and_one(dir[0]))
                info.append(self._to_zero_and_one(dir[1]))
            else:
                info.extend([0.0, 0.0])
        else:
            info.extend([0.0, 0.0])

        cos_beta = heading_dir_now.dot(heading_dir_last) / (norm(*heading_dir_now) * norm(*heading_dir_last))
        beta_diff = np.arccos(clip(cos_beta, 0.0, 1.0))
        yaw_rate = beta_diff / 0.1
        info.append(clip(yaw_rate, 0.0, 1.0))

        long, lateral = vehicle.lane.local_coordinates(vehicle.position)
        info.append(clip((lateral * 2 / vehicle.routing_localization.get_current_lane_width() + 1.0) / 2.0, 0.0, 1.0))
        info.append(clip(long / DISTANCE, 0.0, 1.0))
        return info

    def observe(self, vehicle):
        state = self.observe_ego_state(vehicle)
        other_v_info = []
        if self.config["lidar"]["num_others"] > 0:
            other_v_info += self.overwritten_get_surrounding_vehicles_info(
                lidar=vehicle.lidar, ego_vehicle=vehicle, num_others=self.config["lidar"]["num_others"]
            )
        return np.concatenate((state, np.asarray(other_v_info)))

    def overwritten_get_surrounding_vehicles_info(self, lidar, ego_vehicle, num_others: int = 4):
        # surrounding_vehicles = list(lidar.get_surrounding_vehicles())
        surrounding_vehicles = list(self._env.engine.traffic_manager.vehicles)[1:]
        surrounding_vehicles.sort(
            key=lambda v: norm(ego_vehicle.position[0] - v.position[0], ego_vehicle.position[1] - v.position[1])
        )

        # dis = [(str(v), norm(ego_vehicle.position[0] - v.position[0], ego_vehicle.position[1] - v.position[1]))
        # for v in surrounding_vehicles]

        surrounding_vehicles += [None] * num_others
        res = []
        for vehicle in surrounding_vehicles[:num_others]:
            if vehicle is not None:
                relative_position = ego_vehicle.projection(vehicle.position - ego_vehicle.position)
                dist = norm(relative_position[0], relative_position[1])
                if dist > DISTANCE:
                    if self.config["use_extra_state"]:
                        res += [0.0] * self._traffic_vehicle_state_dim
                    else:
                        res += [0.0] * self._traffic_vehicle_state_dim_wo_extra
                    continue
                relative_velocity = ego_vehicle.projection(vehicle.velocity - ego_vehicle.velocity)
                res.append(clip(dist / DISTANCE, 0.0, 1.0))
                res.append(clip(norm(relative_velocity[0], relative_velocity[1]) / ego_vehicle.max_speed, 0.0, 1.0))
                res.append(clip((relative_position[0] / DISTANCE + 1) / 2, 0.0, 1.0))
                res.append(clip((relative_position[1] / DISTANCE + 1) / 2, 0.0, 1.0))
                res.append(clip((relative_velocity[0] / ego_vehicle.max_speed + 1) / 2, 0.0, 1.0))
                res.append(clip((relative_velocity[1] / ego_vehicle.max_speed + 1) / 2, 0.0, 1.0))

                if self.config["use_extra_state"]:
                    res.extend(self.traffic_vehicle_state(vehicle))
            else:
                if self.config["use_extra_state"]:
                    res += [0.0] * self._traffic_vehicle_state_dim
                else:
                    res += [0.0] * self._traffic_vehicle_state_dim_wo_extra

        # p1 = []
        # p2 = []
        # for vehicle in surrounding_vehicles[:num_others]:
        #     if vehicle is not None:
        #         relative_position = ego_vehicle.projection(vehicle.position - ego_vehicle.position)
        #         relative_velocity = ego_vehicle.projection(vehicle.velocity - ego_vehicle.velocity)
        #         p1.append(relative_position)
        #         p2.append(relative_velocity)
        # print("Detected Others Position: {}, Velocity: {}".format(p1, p2))

        return res

    def traffic_vehicle_state(self, vehicle):
        s = []
        state = vehicle.to_dict()
        s.append(state['vx'] / vehicle.MAX_SPEED)
        s.append(state['vy'] / vehicle.MAX_SPEED)
        s.append(state["cos_h"])
        s.append(state["sin_h"])

        # TODO(pzh): This is stupid here!!
        pm = get_engine().policy_manager
        p = pm.get_policy(vehicle.name)
        # s.append(state["cos_d"])
        # s.append(state["sin_d"])

        # TODO(pzh): This is a workaround!!
        if p is None:
            s.append(0.0)
            s.append(0.0)
            s.append(0.0)
        else:
            s.append(p.destination[0])
            s.append(p.destination[1])
            target_speed = p.target_speed
            s.append(target_speed / vehicle.MAX_SPEED)

        s.append(vehicle.speed / vehicle.MAX_SPEED)
        s.append(math.cos(vehicle.heading))
        s.append(math.sin(vehicle.heading))
        s.append(vehicle.action["steering"])
        s.append(vehicle.action["acceleration"] / vehicle.ACC_MAX)
        ret = []
        for v in s:
            ret.append(self._to_zero_and_one(v))
        return ret

    @staticmethod
    def _to_zero_and_one(v):
        return (clip(v, -1, +1) + 1) / 2


class PGDriveEnvV2Minimal(PGDriveEnvV2):
    @classmethod
    def default_config(cls) -> Config:
        config = super(PGDriveEnvV2Minimal, cls).default_config()
        config.update({"num_others": 4, "use_extra_state": True})
        config["vehicle_config"]["lidar"]["distance"] = 0
        config["vehicle_config"]["lidar"]["num_lasers"] = 0
        config["vehicle_config"]["lidar"]["num_others"] = 0
        config["vehicle_config"]["side_detector"]["num_lasers"] = 0
        config["vehicle_config"]["side_detector"]["distance"] = 0
        config["vehicle_config"]["lane_line_detector"]["num_lasers"] = 0
        config["vehicle_config"]["lane_line_detector"]["distance"] = 0
        return config

    def get_single_observation(self, vehicle_config: "Config") -> "ObservationBase":
        return MinimalObservation(vehicle_config, self)

    def _post_process_config(self, config):
        config = super(PGDriveEnvV2Minimal, self)._post_process_config(config)
        assert config["vehicle_config"]["lidar"]["num_others"] == 0
        assert config["vehicle_config"]["lidar"]["num_lasers"] == 0
        assert config["num_others"] >= 0
        config["vehicle_config"]["lidar"]["num_others"] = config["num_others"]
        config["vehicle_config"]["use_extra_state"] = config["use_extra_state"]
        return config


if __name__ == '__main__':

    def _act(env, action):
        assert env.action_space.contains(action)
        obs, reward, done, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isscalar(reward)
        assert isinstance(info, dict)

    env = PGDriveEnvV2Minimal(
        {
            "use_render": False,
            "fast": True,
            "num_others": 4,
            "map": "SSS",
            "use_extra_state": True,
            "traffic_density": 0.1
        }
    )
    try:
        obs = env.reset()
        assert env.observation_space.contains(obs)
        _act(env, env.action_space.sample())
        env.reset()
        for _ in range(100):
            _act(env, [0, 1])
    finally:
        env.close()
