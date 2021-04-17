import logging

import gym
import numpy as np
from gym.spaces import Box
from pgdrive.envs.multi_agent_pgdrive import MultiAgentPGDrive
from pgdrive.envs.pgdrive_env_v2 import PGDriveEnvV2
from pgdrive.obs import ObservationType
from pgdrive.obs.state_obs import StateObservation
from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.scene_creator.blocks.roundabout import Roundabout
from pgdrive.scene_creator.map import PGMap
from pgdrive.scene_creator.road.road import Road
from pgdrive.scene_manager.spawn_manager import SpawnManager
from pgdrive.scene_manager.target_vehicle_manager import TargetVehicleManager
from pgdrive.utils import get_np_random, norm, PGConfig

MARoundaboutConfig = dict(map_config=dict(exit_length=50, lane_num=2))


class MARoundaboutMap(PGMap):
    def _generate(self, pg_world):
        length = self.config["exit_length"]

        parent_node_path, pg_physics_world = pg_world.worldNP, pg_world.physics_world
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"

        # Build a first-block
        last_block = FirstBlock(
            self.road_network,
            self.config[self.LANE_WIDTH],
            self.config[self.LANE_NUM],
            parent_node_path,
            pg_physics_world,
            1,
            length=length
        )
        self.blocks.append(last_block)

        # Build roundabout
        Roundabout.EXIT_PART_LENGTH = length
        last_block = Roundabout(1, last_block.get_socket(index=0), self.road_network, random_seed=1)
        last_block.construct_block(
            parent_node_path,
            pg_physics_world,
            extra_config={
                "exit_radius": 10,
                "inner_radius": 30,
                "angle": 70,
                # Note: lane_num is set in config.map_config.lane_num
            }
        )
        self.blocks.append(last_block)


class LidarStateObservationMARound(ObservationType):
    def __init__(self, vehicle_config):
        self.state_obs = StateObservation(vehicle_config)
        super(LidarStateObservationMARound, self).__init__(vehicle_config)
        self.state_length = list(self.state_obs.observation_space.shape)[0]

    @property
    def observation_space(self):
        shape = list(self.state_obs.observation_space.shape)
        if self.config["lidar"]["num_lasers"] > 0 and self.config["lidar"]["distance"] > 0:
            # Number of lidar rays and distance should be positive!
            shape[0] += self.config["lidar"]["num_lasers"] + self.config["lidar"]["num_others"] * self.state_length
        return gym.spaces.Box(-0.0, 1.0, shape=tuple(shape), dtype=np.float32)

    def observe(self, vehicle):
        num_others = self.config["lidar"]["num_others"]
        state = self.state_observe(vehicle)
        other_v_info = []
        if vehicle.lidar is not None:
            if self.config["lidar"]["num_others"] > 0:
                surrounding_vehicles = list(vehicle.lidar.get_surrounding_vehicles())
                surrounding_vehicles.sort(
                    key=lambda v: norm(vehicle.position[0] - v.position[0], vehicle.position[1] - v.position[1])
                )
                surrounding_vehicles += [None] * num_others
                for tmp_v in surrounding_vehicles[:num_others]:
                    if tmp_v is not None:
                        tmp_v = tmp_v.get_vehicle()
                        other_v_info += self.state_observe(tmp_v).tolist()
                    else:
                        other_v_info += [0] * self.state_length
            other_v_info += self._add_noise_to_cloud_points(
                vehicle.lidar.get_cloud_points(),
                gaussian_noise=self.config["lidar"]["gaussian_noise"],
                dropout_prob=self.config["lidar"]["dropout_prob"]
            )
        return np.concatenate((state, np.asarray(other_v_info)))

    def state_observe(self, vehicle):
        return self.state_obs.observe(vehicle)

    def _add_noise_to_cloud_points(self, points, gaussian_noise, dropout_prob):
        if gaussian_noise > 0.0:
            points = np.asarray(points)
            points = np.clip(points + np.random.normal(loc=0.0, scale=gaussian_noise, size=points.shape), 0.0, 1.0)

        if dropout_prob > 0.0:
            assert dropout_prob <= 1.0
            points = np.asarray(points)
            points[np.random.uniform(0, 1, size=points.shape) < dropout_prob] = 0.0

        return list(points)


class MultiAgentRoundaboutEnv(MultiAgentPGDrive):
    _DEBUG_RANDOM_SEED = None
    spawn_roads = [
        Road(FirstBlock.NODE_2, FirstBlock.NODE_3),
        -Road(Roundabout.node(1, 0, 2), Roundabout.node(1, 0, 3)),
        -Road(Roundabout.node(1, 1, 2), Roundabout.node(1, 1, 3)),
        -Road(Roundabout.node(1, 2, 2), Roundabout.node(1, 2, 3)),
    ]

    @staticmethod
    def default_config() -> PGConfig:
        return MultiAgentPGDrive.default_config().update(MARoundaboutConfig, allow_overwrite=True)

    def __init__(self, config=None):
        super(MultiAgentRoundaboutEnv, self).__init__(config)
        self.target_vehicle_manager = TargetVehicleManager()

    def _update_map(self, episode_data: dict = None, force_seed=None):
        if episode_data is not None:
            raise ValueError()
        map_config = self.config["map_config"]
        map_config.update({"seed": self.current_seed})

        if self.current_map is None:
            self.current_seed = 0
            new_map = MARoundaboutMap(self.pg_world, map_config)
            self.maps[self.current_seed] = new_map
            self.current_map = self.maps[self.current_seed]

    def _after_lazy_init(self):
        super(MultiAgentRoundaboutEnv, self)._after_lazy_init()

        # Use top-down view by default
        if hasattr(self, "main_camera") and self.main_camera is not None:
            bird_camera_height = self.config["bird_camera_height"]
            self.main_camera.camera.setPos(0, 0, bird_camera_height)
            self.main_camera.bird_camera_height = bird_camera_height
            self.main_camera.stop_chase(self.pg_world)
            # self.main_camera.camera.setPos(300, 20, bird_camera_height)
            self.main_camera.camera_x += 95
            self.main_camera.camera_y += 15

    def _process_extra_config(self, config):
        config = super(MultiAgentRoundaboutEnv, self)._process_extra_config(config)
        config = self._update_agent_pos_configs(config)
        return super(MultiAgentRoundaboutEnv, self)._process_extra_config(config)

    def _update_agent_pos_configs(self, config):
        self._spawn_manager = SpawnManager(
            spawn_roads=self.spawn_roads,
            exit_length=config["map_config"]["exit_length"],
            lane_num=config["map_config"]["lane_num"],
            num_agents=config["num_agents"],
            vehicle_config=config["vehicle_config"]
        )
        config["target_vehicle_configs"] = self._spawn_manager.get_target_vehicle_configs(
            config["num_agents"], seed=self._DEBUG_RANDOM_SEED
        )
        return config

    def reset(self, *args, **kwargs):
        # Shuffle spawn places!
        self.config = self._update_agent_pos_configs(self.config)

        for v in self.done_vehicles.values():
            v.chassis_np.node().setStatic(False)

        # Multi-agent related reset
        # avoid create new observation!
        obses = self.target_vehicle_manager.get_observations() or list(self.observations.values())
        assert len(obses) == len(self.config["target_vehicle_configs"].keys())
        self.observations = {k: v for k, v in zip(self.config["target_vehicle_configs"].keys(), obses)}
        self.done_observations = dict()

        # Must change in-place!
        obs_spaces = self.target_vehicle_manager.get_observation_spaces() or list(
            self.observation_space.spaces.values()
        )
        assert len(obs_spaces) == len(self.config["target_vehicle_configs"].keys())
        for o in obs_spaces:
            assert isinstance(o, Box)
        self.observation_space.spaces = {k: v for k, v in zip(self.observations.keys(), obs_spaces)}
        action_spaces = self.target_vehicle_manager.get_action_spaces() or list(self.action_space.spaces.values())
        self.action_space.spaces = {k: v for k, v in zip(self.observations.keys(), action_spaces)}

        ret = PGDriveEnvV2.reset(self, *args, **kwargs)

        assert len(self.vehicles) == self.num_agents
        self.for_each_vehicle(self._update_destination_for)

        self.target_vehicle_manager.reset(
            vehicles=self.vehicles,
            observation_spaces=self.observation_space.spaces,
            observations=self.observations,
            action_spaces=self.action_space.spaces
        )
        return ret

    def step(self, actions):
        o, r, d, i = super(MultiAgentRoundaboutEnv, self).step(actions)

        # Update respawn manager
        if self.episode_steps >= self.config["horizon"]:
            self.target_vehicle_manager.set_allow_respawn(False)
        self._spawn_manager.update(self.vehicles, self.current_map)
        new_obs_dict = self._respawn()
        if new_obs_dict:
            for new_id, new_obs in new_obs_dict.items():
                o[new_id] = new_obs
                r[new_id] = 0.0
                i[new_id] = {}
                d[new_id] = False

        # Update __all__
        d["__all__"] = (
            ((self.episode_steps >= self.config["horizon"]) and (all(d.values()))) or (len(self.vehicles) == 0)
            or (self.episode_steps >= 5 * self.config["horizon"])
        )
        if d["__all__"]:
            for k in d.keys():
                d[k] = True

        return o, r, d, i

    def _update_destination_for(self, vehicle):
        # when agent re-joined to the game, call this to set the new route to destination
        end_road = -get_np_random(self._DEBUG_RANDOM_SEED).choice(self.spawn_roads)  # Use negative road!
        vehicle.routing_localization.set_route(vehicle.lane_index[0], end_road.end_node)

    def _respawn(self):
        new_obs_dict = {}
        while True:
            new_id, new_obs = self._respawn_single_vehicle()
            if new_obs is not None:
                new_obs_dict[new_id] = new_obs
            else:
                break
        return new_obs_dict

    def _force_respawn(self, agent_name):
        """
        This function can force a given vehicle to respawn!
        """
        self.target_vehicle_manager.finish(agent_name)
        new_id, new_obs = self._respawn_single_vehicle()
        return new_id, new_obs

    def _respawn_single_vehicle(self):
        """
        Arbitrary insert a new vehicle to a new spawn place if possible.
        """
        allow_respawn, vehicle_info = self.target_vehicle_manager.propose_new_vehicle()
        if vehicle_info is None:  # No more vehicle to be assigned.
            return None, None
        if not allow_respawn:
            # If not allow to respawn, move agents to some rural places.
            v = vehicle_info["vehicle"]
            v.set_position((-999, -999))
            v.set_static(True)
            self.target_vehicle_manager.confirm_respawn(False, vehicle_info)
            return None, None
        v = vehicle_info["vehicle"]
        dead_vehicle_id = vehicle_info["old_name"]
        bp_index = self._replace_vehicles(v)
        if bp_index is None:  # No more spawn places to be assigned.
            self.target_vehicle_manager.confirm_respawn(False, vehicle_info)
            return None, None

        self.target_vehicle_manager.confirm_respawn(True, vehicle_info)

        new_id = vehicle_info["new_name"]
        v.set_static(False)
        self.vehicles[new_id] = v  # Put it to new vehicle id.
        self.dones[new_id] = False  # Put it in the internal dead-tracking dict.
        self._spawn_manager.confirm_respawn(spawn_place_id=bp_index, vehicle_id=new_id)
        logging.debug("{} Dead. {} Respawn!".format(dead_vehicle_id, new_id))

        self.observations[new_id] = vehicle_info["observation"]
        self.observations[new_id].reset(self, v)
        self.observation_space.spaces[new_id] = vehicle_info["observation_space"]
        self.action_space.spaces[new_id] = vehicle_info["action_space"]
        new_obs = self.observations[new_id].observe(v)
        return new_id, new_obs

    def _after_vehicle_done(self, obs=None, reward=None, dones: dict = None, info=None):
        for v_id, v_info in info.items():
            if v_info.get("episode_length", 0) >= self.config["horizon"]:
                if dones[v_id] is not None:
                    info[v_id]["max_step"] = True
                    dones[v_id] = True
                    self.dones[v_id] = True
        for dead_vehicle_id, done in dones.items():
            if done:
                self.target_vehicle_manager.finish(dead_vehicle_id)
                self.vehicles.pop(dead_vehicle_id)
                self.action_space.spaces.pop(dead_vehicle_id)
        return obs, reward, dones, info

    def _reset_vehicles(self):
        vehicles = self.target_vehicle_manager.get_vehicle_list() or list(self.vehicles.values())
        assert len(vehicles) == len(self.observations)
        self.vehicles = {k: v for k, v in zip(self.observations.keys(), vehicles)}
        self.done_vehicles = {}

        # update config (for new possible spawn places)
        for v_id, v in self.vehicles.items():
            v.vehicle_config = self._get_target_vehicle_config(self.config["target_vehicle_configs"][v_id])

        # reset
        self.for_each_vehicle(lambda v: v.reset(self.current_map))

    def _replace_vehicles(self, v):
        v.prepare_step([0, -1])
        safe_places_dict = self._spawn_manager.get_available_spawn_places()
        if len(safe_places_dict) == 0:
            # No more run, just wait!
            return None
        assert len(safe_places_dict) > 0
        bp_index = get_np_random(self._DEBUG_RANDOM_SEED).choice(list(safe_places_dict.keys()), 1)[0]
        new_spawn_place = safe_places_dict[bp_index]
        new_spawn_place_config = new_spawn_place["config"]
        v.vehicle_config.update(new_spawn_place_config)
        v.reset(self.current_map)
        self._update_destination_for(v)
        v.update_state(detector_mask=None)
        return bp_index

    def get_single_observation(self, vehicle_config: "PGConfig") -> "ObservationType":
        return LidarStateObservationMARound(vehicle_config)


def _draw():
    env = MultiAgentRoundaboutEnv()
    o = env.reset()
    from pgdrive.utils import draw_top_down_map
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
                "use_saver": True,
                "save_level": 1.
            },
            "pg_world_config": {
                "debug_physics_world": True
            },
            "fast": True,
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
                    i, total_r, total_r / env.target_vehicle_manager.next_agent_count
                )
            )
            break
        if len(env.vehicles) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()


def _vis():
    env = MultiAgentRoundaboutEnv(
        {
            "horizon": 100,
            "vehicle_config": {
                "lidar": {
                    "num_lasers": 72,
                    "num_others": 0,
                    "distance": 40
                },
                "show_lidar": True,
            },
            "fast": True,
            "use_render": True,
            "debug": True,
            "manual_control": True,
            "num_agents": 8,
        }
    )
    o = env.reset()
    total_r = 0
    ep_s = 0
    for i in range(1, 100000):
        o, r, d, info = env.step({k: [0.0, 1.0] for k in env.vehicles.keys()})
        for r_ in r.values():
            total_r += r_
        ep_s += 1
        # d.update({"total_r": total_r, "episode length": ep_s})
        render_text = {
            "total_r": total_r,
            "episode length": ep_s,
            "cam_x": env.main_camera.camera_x,
            "cam_y": env.main_camera.camera_y,
            "cam_z": env.main_camera.bird_camera_height
        }
        env.render(text=render_text)
        if d["__all__"]:
            print(
                "Finish! Current step {}. Group Reward: {}. Average reward: {}".format(
                    i, total_r, total_r / env.target_vehicle_manager.next_agent_count
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
    env = MultiAgentRoundaboutEnv({"num_agents": 16})
    obs = env.reset()
    start = time.time()
    for s in range(10000):
        o, r, d, i = env.step(env.action_space.sample())

        # mask_ratio = env.scene_manager.detector_mask.get_mask_ratio()
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
    print(f"(PGDriveEnvV2) Total Time Elapse: {time.time() - start}")


def _long_run():
    # Please refer to test_ma_roundabout_reward_done_alignment()
    _out_of_road_penalty = 3
    env = MultiAgentRoundaboutEnv(
        {
            "num_agents": 32,
            "vehicle_config": {
                "lidar": {
                    "num_others": 8
                }
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
    # _profile()
    # _long_run()
