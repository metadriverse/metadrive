import copy

import gym
import numpy as np

from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.roundabout import Roundabout
from metadrive.component.road_network import Road
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.map_manager import PGMapManager
from metadrive.manager.spawn_manager import SpawnManager
from metadrive.obs.observation_base import ObservationBase
from metadrive.obs.state_obs import StateObservation
from metadrive.utils import norm, Config

MARoundaboutConfig = dict(
    spawn_roads=[
        Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3),
        -Road(Roundabout.node(1, 0, 2), Roundabout.node(1, 0, 3)),
        -Road(Roundabout.node(1, 1, 2), Roundabout.node(1, 1, 3)),
        -Road(Roundabout.node(1, 2, 2), Roundabout.node(1, 2, 3)),
    ],
    map_config=dict(exit_length=60, lane_num=2),
    top_down_camera_initial_x=95,
    top_down_camera_initial_y=15,
    top_down_camera_initial_z=120,
    num_agents=40,
)


class MARoundaboutMap(PGMap):
    def _generate(self):
        length = self.config["exit_length"]

        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"

        # Build a first-block
        last_block = FirstPGBlock(
            self.road_network,
            self.config[self.LANE_WIDTH],
            self.config[self.LANE_NUM],
            parent_node_path,
            physics_world,
            length=length
        )
        self.blocks.append(last_block)

        # Build roundabout
        Roundabout.EXIT_PART_LENGTH = length
        last_block = Roundabout(
            1, last_block.get_socket(index=0), self.road_network, random_seed=1, ignore_intersection_checking=False
        )
        last_block.construct_block(
            parent_node_path,
            physics_world,
            extra_config={
                "exit_radius": 10,
                "inner_radius": 30,
                "angle": 70,
                # Note: lane_num is set in config.map_config.lane_num
            }
        )
        self.blocks.append(last_block)


class MARoundaboutPGMapManager(PGMapManager):
    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(MARoundaboutMap, map_config=config["map_config"], random_seed=None)
        else:
            assert len(self.spawned_objects) == 1, "It is supposed to contain one map in this manager"
            _map = self.spawned_objects.values()[0]
        self.load_map(_map)
        self.current_map.spawn_roads = config["spawn_roads"]


# Note: This class allow one agent accessing its neighbors' observation.
# But we never use this functionality in formal experiment. Therefore we just deprecate it for
# clarity!
# class LidarStateObservationMARound(ObservationBase):
#     def __init__(self, vehicle_config):
#         self.state_obs = StateObservation(vehicle_config)
#         super(LidarStateObservationMARound, self).__init__(vehicle_config)
#         self.state_length = list(self.state_obs.observation_space.shape)[0]
#         self.cloud_points = None
#         self.detected_objects = None
#
#     @property
#     def observation_space(self):
#         shape = list(self.state_obs.observation_space.shape)
#         if self.config["lidar"]["num_lasers"] > 0 and self.config["lidar"]["distance"] > 0:
#             # Number of lidar rays and distance should be positive!
#             shape[0] += self.config["lidar"]["num_lasers"] + self.config["lidar"]["num_others"] * self.state_length
#         return gym.spaces.Box(-0.0, 1.0, shape=tuple(shape), dtype=np.float32)
#
#     def observe(self, vehicle):
#         num_others = self.config["lidar"]["num_others"]
#         state = self.state_observe(vehicle)
#         other_v_info = []
#         if vehicle.lidar.available:
#             cloud_points, detected_objects = vehicle.lidar.perceive(vehicle)
#             if self.config["lidar"]["num_others"] > 0:
#                 surrounding_vehicles = list(vehicle.lidar.get_surrounding_vehicles(detected_objects))
#                 surrounding_vehicles.sort(
#                     key=lambda v: norm(vehicle.position[0] - v.position[0], vehicle.position[1] - v.position[1])
#                 )
#                 surrounding_vehicles += [None] * num_others
#                 for tmp_v in surrounding_vehicles[:num_others]:
#                     if tmp_v is not None:
#                         tmp_v = tmp_v
#                         other_v_info += self.state_observe(tmp_v).tolist()
#                     else:
#                         other_v_info += [0] * self.state_length
#             other_v_info += self._add_noise_to_cloud_points(
#                 cloud_points,
#                 gaussian_noise=self.config["lidar"]["gaussian_noise"],
#                 dropout_prob=self.config["lidar"]["dropout_prob"]
#             )
#             self.cloud_points = cloud_points
#             self.detected_objects = detected_objects
#         self.current_observation = np.concatenate((state, np.asarray(other_v_info)))
#         return self.current_observation.astype(np.float32)
#
#     def state_observe(self, vehicle):
#         return self.state_obs.observe(vehicle)
#
#     def _add_noise_to_cloud_points(self, points, gaussian_noise, dropout_prob):
#         if gaussian_noise > 0.0:
#             points = np.asarray(points)
#             points = np.clip(points + np.random.normal(loc=0.0, scale=gaussian_noise, size=points.shape), 0.0, 1.0)
#
#         if dropout_prob > 0.0:
#             assert dropout_prob <= 1.0
#             points = np.asarray(points)
#             points[np.random.uniform(0, 1, size=points.shape) < dropout_prob] = 0.0
#
#         return list(points)


class RoundaboutSpawnManager(SpawnManager):
    def update_destination_for(self, vehicle_id, vehicle_config):
        end_roads = copy.deepcopy(self.engine.global_config["spawn_roads"])
        end_road = -self.np_random.choice(end_roads)  # Use negative road!
        vehicle_config["destination"] = end_road.end_node
        return vehicle_config


class MultiAgentRoundaboutEnv(MultiAgentMetaDrive):
    @staticmethod
    def default_config() -> Config:
        return MultiAgentMetaDrive.default_config().update(MARoundaboutConfig, allow_add_new_key=True)

    def setup_engine(self):
        super(MultiAgentRoundaboutEnv, self).setup_engine()
        self.engine.update_manager("spawn_manager", RoundaboutSpawnManager())
        self.engine.update_manager("map_manager", MARoundaboutPGMapManager())


def _draw():
    env = MultiAgentRoundaboutEnv()
    o = env.reset()
    from metadrive.utils.draw_top_down_map import draw_top_down_map
    import matplotlib.pyplot as plt

    plt.imshow(draw_top_down_map(env.current_map))
    plt.show()
    env.close()


def _expert():
    env = MultiAgentRoundaboutEnv(
        {
            "vehicle_config": {
                "lidar": {
                    "num_lasers": 240,
                    "num_others": 4,
                    "distance": 50
                },
            },
            "use_AI_protector": True,
            "save_level": 1.,
            "debug_physics_world": True,

            # "use_render": True,
            "debug": True,
            "manual_control": True,
            "num_agents": 4,
        }
    )
    o = env.reset()
    total_r = 0
    ep_s = 0
    for i in range(1, 100000):
        o, r, d, info = env.step(env.action_space.sample())
        for r_ in r.values():
            total_r += r_
        ep_s += 1
        d.update({"total_r": total_r, "episode length": ep_s})
        # env.render(text=d)
        if d["__all__"]:
            print(
                "Finish! Current step {}. Group Reward: {}. Average reward: {}".format(
                    i, total_r, total_r / env.agent_manager.next_agent_count
                )
            )
            break
        if len(env.vehicles) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()


def _vis_debug_respawn():
    env = MultiAgentRoundaboutEnv(
        {
            "horizon": 100000,
            "vehicle_config": {
                "lidar": {
                    "num_lasers": 72,
                    "num_others": 0,
                    "distance": 40
                },
                "show_lidar": False,
            },
            "debug_physics_world": True,
            "use_render": True,
            "debug": False,
            "manual_control": True,
            "num_agents": 40,
        }
    )
    o = env.reset()
    total_r = 0
    ep_s = 0
    for i in range(1, 100000):
        action = {k: [0.0, .0] for k in env.vehicles.keys()}
        o, r, d, info = env.step(action)
        for r_ in r.values():
            total_r += r_
        ep_s += 1
        # d.update({"total_r": total_r, "episode length": ep_s})
        render_text = {
            "total_r": total_r,
            "episode length": ep_s,
            "cam_x": env.main_camera.camera_x,
            "cam_y": env.main_camera.camera_y,
            "cam_z": env.main_camera.top_down_camera_height
        }
        env.render(text=render_text)
        if d["__all__"]:
            print(
                "Finish! Current step {}. Group Reward: {}. Average reward: {}".format(
                    i, total_r, total_r / env.agent_manager.next_agent_count
                )
            )
            # break
        if len(env.vehicles) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()


def _vis():
    env = MultiAgentRoundaboutEnv(
        {
            "horizon": 100000,
            "vehicle_config": {
                "lidar": {
                    "num_lasers": 72,
                    "num_others": 0,
                    "distance": 40
                },
                "show_lidar": False,
            },
            "use_render": True,
            "debug": False,
            "manual_control": True,
            "num_agents": -1,
        }
    )
    o = env.reset()
    total_r = 0
    ep_s = 0
    for i in range(1, 100000):
        o, r, d, info = env.step({k: [0, .0] for k in env.vehicles.keys()})
        for r_ in r.values():
            total_r += r_
        ep_s += 1
        # d.update({"total_r": total_r, "episode length": ep_s})
        render_text = {
            "total_r": total_r,
            "episode length": ep_s,
            "cam_x": env.main_camera.camera_x,
            "cam_y": env.main_camera.camera_y,
            "cam_z": env.main_camera.top_down_camera_height
        }
        env.render(text=render_text)
        if d["__all__"]:
            print(
                "Finish! Current step {}. Group Reward: {}. Average reward: {}".format(
                    i, total_r, total_r / env.agent_manager.next_agent_count
                )
            )
            # break
        if len(env.vehicles) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()


def _profile():
    import time
    env = MultiAgentRoundaboutEnv({"num_agents": 40})
    obs = env.reset()
    start = time.time()
    for s in range(10000):
        o, r, d, i = env.step(env.action_space.sample())

        # mask_ratio = env.engine.detector_mask.get_mask_ratio()
        # print("Mask ratio: ", mask_ratio)

        if all(d.values()):
            env.reset()
        if (s + 1) % 100 == 0:
            print(
                "Finish {}/10000 simulation steps. Time elapse: {:.4f}. Average FPS: {:.4f}".format(
                    s + 1,
                    time.time() - start, (s + 1) / (time.time() - start)
                )
            )
    print(f"(MetaDriveEnv) Total Time Elapse: {time.time() - start}")


def _long_run():
    # Please refer to test_ma_roundabout_reward_done_alignment()
    _out_of_road_penalty = 3
    env = MultiAgentRoundaboutEnv(
        {
            "num_agents": 40,
            "vehicle_config": {
                "lidar": {
                    "num_others": 8
                },
            },
            **dict(
                out_of_road_penalty=_out_of_road_penalty,
                crash_vehicle_penalty=1.333,
                crash_object_penalty=11,
                crash_vehicle_cost=13,
                crash_object_cost=17,
                out_of_road_cost=19,
            )
        }
    )
    try:
        obs = env.reset()
        assert env.observation_space.contains(obs)
        for step in range(10000):
            act = env.action_space.sample()
            o, r, d, i = env.step(act)
            if step == 0:
                assert not any(d.values())

            if any(d.values()):
                print("Current Done: {}\nReward: {}".format(d, r))
                for kkk, ddd in d.items():
                    if ddd and kkk != "__all__":
                        print("Info {}: {}\n".format(kkk, i[kkk]))
                print("\n")

            for kkk, rrr in r.items():
                if rrr == -_out_of_road_penalty:
                    assert d[kkk]

            if (step + 1) % 200 == 0:
                print(
                    "{}/{} Agents: {} {}\nO: {}\nR: {}\nD: {}\nI: {}\n\n".format(
                        step + 1, 10000, len(env.vehicles), list(env.vehicles.keys()),
                        {k: (oo.shape, oo.mean(), oo.min(), oo.max())
                         for k, oo in o.items()}, r, d, i
                    )
                )
            if d["__all__"]:
                print('Current step: ', step)
                break
    finally:
        env.close()


if __name__ == "__main__":
    # _draw()
    _vis()
    # _vis_debug_respawn()
    # _profile()
    # _long_run()
    # pygame_replay("round", MultiAgentRoundaboutEnv)
