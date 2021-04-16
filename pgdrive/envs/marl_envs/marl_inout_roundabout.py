import copy
import logging
from math import floor

import numpy as np
from gym.spaces import Box

from pgdrive.envs import PGDriveEnvV2
from pgdrive.envs.multi_agent_pgdrive import MultiAgentPGDrive
from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.scene_creator.blocks.roundabout import Roundabout
from pgdrive.scene_creator.map import PGMap
from pgdrive.scene_creator.road.road import Road
from pgdrive.utils import get_np_random, PGConfig, distance_greater

MARoundaboutConfig = {
    "num_agents": 2,  # Number of maximum agents in the scenarios.
    "horizon": 1000,  # We will stop reborn vehicles after this timesteps.

    # Vehicle
    "vehicle_config": {
        "lidar": {
            "num_lasers": 72,
            "distance": 40,
            "num_others": 0,
        },
        "born_longitude": 5,
        "born_lateral": 0,
    },

    # Map
    "map_config": {
        "exit_length": 50,
        "lane_num": 2
    },

    # Reward scheme
    "crash_done": False,
    "out_of_road_penalty": 5.0,
    "crash_vehicle_penalty": 1.0,
    "crash_object_penalty": 1.0,
    "auto_termination": False,
    "camera_height": 4,
}


class TargetVehicleManager:
    """
    vehicle name: unique name for each vehicle instance, random string.
    agent name: agent name that exists in the environment, like agent0, agent1, ....
    """
    def __init__(self, ):
        self.agent_to_vehicle = {}
        self.vehicle_to_agent = {}
        self.pending_vehicles = {}
        self.active_vehicles = {}
        self.next_agent_count = 0
        self.observations = {}
        self.observation_spaces = {}
        self.action_spaces = {}
        self.allow_reborn = True

    def reset(self, vehicles, observation_spaces, action_spaces, observations):
        self.agent_to_vehicle = {k: v.name for k, v in vehicles.items()}
        self.vehicle_to_agent = {v.name: k for k, v in vehicles.items()}
        self.active_vehicles = {v.name: v for v in vehicles.values()}
        self.next_agent_count = len(vehicles)
        self.observations = {vehicles[k].name: v for k, v in observations.items()}
        self.observation_spaces = {vehicles[k].name: v for k, v in observation_spaces.items()}
        for o in observation_spaces.values():
            assert isinstance(o, Box)
        self.action_spaces = {vehicles[k].name: v for k, v in action_spaces.items()}
        for o in action_spaces.values():
            assert isinstance(o, Box)
        self.pending_vehicles = {}
        self.allow_reborn = True

    def finish(self, agent_name):
        vehicle_name = self.agent_to_vehicle[agent_name]
        v = self.active_vehicles.pop(vehicle_name)
        assert vehicle_name not in self.active_vehicles
        self.pending_vehicles[vehicle_name] = v
        self._check()

    def _check(self):
        current_keys = sorted(list(self.pending_vehicles.keys()) + list(self.active_vehicles.keys()))
        exist_keys = sorted(list(self.vehicle_to_agent.keys()))
        assert current_keys == exist_keys, "You should confirm_reborn() after request for propose_new_vehicle()!"

    def propose_new_vehicle(self):
        self._check()
        if len(self.pending_vehicles) > 0:
            v_id = list(self.pending_vehicles.keys())[0]
            self._check()
            v = self.pending_vehicles.pop(v_id)
            return self.allow_reborn, dict(
                vehicle=v,
                observation=self.observations[v_id],
                observation_space=self.observation_spaces[v_id],
                action_space=self.action_spaces[v_id],
                old_name=self.vehicle_to_agent[v_id],
                new_name="agent{}".format(self.next_agent_count)
            )
        return None, None

    def confirm_reborn(self, success: bool, vehicle_info):
        vehicle = vehicle_info['vehicle']
        if success:
            self.next_agent_count += 1
            self.active_vehicles[vehicle.name] = vehicle
            self.vehicle_to_agent[vehicle.name] = vehicle_info["new_name"]
            self.agent_to_vehicle.pop(vehicle_info["old_name"])
            self.agent_to_vehicle[vehicle_info["new_name"]] = vehicle.name
        else:
            self.pending_vehicles[vehicle.name] = vehicle
        self._check()

    def set_allow_reborn(self, flag: bool):
        self.allow_reborn = flag

    def _translate(self, d):
        return {self.vehicle_to_agent[k]: v for k, v in d.items()}

    def get_vehicle_list(self):
        return list(self.active_vehicles.values()) + list(self.pending_vehicles.values())

    def get_observations(self):
        return list(self.observations.values())

    def get_observation_spaces(self):
        return list(self.observation_spaces.values())

    def get_action_spaces(self):
        return list(self.action_spaces.values())


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


class BornPlaceManager:
    def __init__(self, born_roads, exit_length, lane_num, num_agents, vehicle_config):
        interval = 10
        num_slots = int(floor(exit_length / interval))
        interval = exit_length / num_slots
        assert num_agents <= lane_num * len(born_roads) * num_slots, (
            "Too many agents! We only accepet {} agents, but you have {} agents!".format(
                lane_num * len(born_roads) * num_slots, num_agents
            )
        )

        # We can spawn agents in the middle of road at the initial time, but when some vehicles need to be reborn,
        # then we have to set it to the farthest places to ensure safety (otherwise the new vehicles may suddenly
        # appear at the middle of the road!)
        target_vehicle_configs = []
        safe_born_places = []
        for i, road in enumerate(born_roads):
            for lane_idx in range(lane_num):
                for j in range(num_slots):
                    long = j * interval + np.random.uniform(0, 0.5 * interval)
                    lane_tuple = road.lane_index(lane_idx)  # like (>>>, 1C0_0_, 1) and so on.
                    target_vehicle_configs.append(
                        dict(
                            identifier="|".join((str(s) for s in lane_tuple + (j, ))),
                            config={
                                "born_lane_index": lane_tuple,
                                "born_longitude": long,
                                "born_lateral": vehicle_config["born_lateral"]
                            }
                        )
                    )
                    if j == 0:
                        safe_born_places.append(target_vehicle_configs[-1].copy())
        self.target_vehicle_configs = target_vehicle_configs
        self.safe_born_places = {v["identifier"]: v for v in safe_born_places}
        self.mapping = {i: set() for i in self.safe_born_places.keys()}
        self.need_update_born_places = True

    def get_target_vehicle_configs(self, num_agents, seed=None):
        target_agents = get_np_random(seed).choice(
            [i for i in range(len(self.target_vehicle_configs))], num_agents, replace=False
        )

        # for rllib compatibility
        ret = {}
        if len(target_agents) > 1:
            for real_idx, idx in enumerate(target_agents):
                v_config = self.target_vehicle_configs[idx]["config"]
                ret["agent{}".format(real_idx)] = v_config
        else:
            ret["agent0"] = self.target_vehicle_configs[0]["config"]
        return copy.deepcopy(ret)

    def update(self, vehicles: dict, map):
        if self.need_update_born_places:
            self.need_update_born_places = False
            for bid, bp in self.safe_born_places.items():
                lane = map.road_network.get_lane(bp["config"]["born_lane_index"])
                self.safe_born_places[bid]["position"] = lane.position(
                    longitudinal=bp["config"]["born_longitude"], lateral=bp["config"]["born_lateral"]
                )
                for vid in vehicles.keys():
                    self.confirm_reborn(bid, vid)  # Just assume everyone is all in the same born place at t=0.

        for bid, vid_set in self.mapping.items():
            removes = []
            for vid in vid_set:
                if (vid not in vehicles) or (distance_greater(self.safe_born_places[bid]["position"],
                                                              vehicles[vid].position, length=10)):
                    removes.append(vid)
            for vid in removes:
                self.mapping[bid].remove(vid)

    def confirm_reborn(self, born_place_id, vehicle_id):
        self.mapping[born_place_id].add(vehicle_id)

    def get_available_born_places(self):
        ret = {}
        for bid in self.safe_born_places.keys():
            if not self.mapping[bid]:  # empty
                ret[bid] = self.safe_born_places[bid]
        return ret


class MultiAgentRoundaboutEnv(MultiAgentPGDrive):
    _DEBUG_RANDOM_SEED = None
    born_roads = [
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
            bird_camera_height = 160
            self.main_camera.camera.setPos(0, 0, bird_camera_height)
            self.main_camera.bird_camera_height = bird_camera_height
            self.main_camera.stop_chase(self.pg_world)
            # self.main_camera.camera.setPos(300, 20, bird_camera_height)
            self.main_camera.camera_x += 100
            self.main_camera.camera_y += 20

    def _process_extra_config(self, config):
        config = super(MultiAgentRoundaboutEnv, self)._process_extra_config(config)
        config = self._update_agent_pos_configs(config)
        return super(MultiAgentRoundaboutEnv, self)._process_extra_config(config)

    def _update_agent_pos_configs(self, config):
        self._born_places_manager = BornPlaceManager(
            born_roads=self.born_roads,
            exit_length=config["map_config"]["exit_length"],
            lane_num=config["map_config"]["lane_num"],
            num_agents=config["num_agents"],
            vehicle_config=config["vehicle_config"]
        )
        config["target_vehicle_configs"] = self._born_places_manager.get_target_vehicle_configs(
            config["num_agents"], seed=self._DEBUG_RANDOM_SEED
        )
        return config

    def reset(self, *args, **kwargs):
        # Shuffle born places!
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

        # Check return alignment
        # o_set_1 = set(kkk for kkk, rrr in r.items() if rrr == -self.config["out_of_road_penalty"])
        # o_set_2 = set(kkk for kkk, iii in i.items() if iii.get("out_of_road"))
        # condition = o_set_1 == o_set_2
        # condition = set(kkk for kkk, rrr in r.items() if rrr == self.config["success_reward"]) == \
        #             set(kkk for kkk, iii in i.items() if iii.get("arrive_dest")) and condition
        # condition = (
        #                     not self.config["crash_done"] or (
        #                     set(kkk for kkk, rrr in r.items() if rrr == -self.config["crash_vehicle_penalty"])
        #                     == set(kkk for kkk, iii in i.items() if iii.get("crash_vehicle"))
        #             )
        #             ) and condition
        # if not condition:
        #     raise ValueError("Observation not aligned!")

        # Update reborn manager
        if self.episode_steps >= self.config["horizon"]:
            self.target_vehicle_manager.set_allow_reborn(False)
        self._born_places_manager.update(self.vehicles, self.current_map)
        new_obs_dict = self._reborn()
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
        end_road = -get_np_random(self._DEBUG_RANDOM_SEED).choice(self.born_roads)  # Use negative road!
        vehicle.routing_localization.set_route(vehicle.lane_index[0], end_road.end_node)

    def _reborn(self):
        new_obs_dict = {}
        while True:
            allow_reborn, vehicle_info = self.target_vehicle_manager.propose_new_vehicle()
            if vehicle_info is None:  # No more vehicle to be assigned.
                break
            if not allow_reborn:
                # If not allow to reborn, move agents to some rural places.
                v = vehicle_info["vehicle"]
                v.set_position((-999, -999))
                v.set_static(True)
                self.target_vehicle_manager.confirm_reborn(False, vehicle_info)
                break
            v = vehicle_info["vehicle"]
            dead_vehicle_id = vehicle_info["old_name"]
            bp_index = self._replace_vehicles(v)
            if bp_index is None:  # No more born places to be assigned.
                self.target_vehicle_manager.confirm_reborn(False, vehicle_info)
                break

            self.target_vehicle_manager.confirm_reborn(True, vehicle_info)

            new_id = vehicle_info["new_name"]
            self.vehicles[new_id] = v  # Put it to new vehicle id.
            self.dones[new_id] = False  # Put it in the internal dead-tracking dict.
            self._born_places_manager.confirm_reborn(born_place_id=bp_index, vehicle_id=new_id)
            logging.debug("{} Dead. {} Reborn!".format(dead_vehicle_id, new_id))

            self.observations[new_id] = vehicle_info["observation"]
            self.observations[new_id].reset(self, v)
            self.observation_space.spaces[new_id] = vehicle_info["observation_space"]
            self.action_space.spaces[new_id] = vehicle_info["action_space"]
            new_obs = self.observations[new_id].observe(v)
            new_obs_dict[new_id] = new_obs
        return new_obs_dict

    def _after_vehicle_done(self, obs=None, reward=None, dones: dict = None, info=None):
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

        # update config (for new possible born places)
        for v_id, v in self.vehicles.items():
            v.vehicle_config = self._get_target_vehicle_config(self.config["target_vehicle_configs"][v_id])

        # reset
        self.for_each_vehicle(lambda v: v.reset(self.current_map))

    def _replace_vehicles(self, v):
        v.prepare_step([0, -1])
        safe_places_dict = self._born_places_manager.get_available_born_places()
        if len(safe_places_dict) == 0:
            # No more run, just wait!
            return None
        assert len(safe_places_dict) > 0
        bp_index = get_np_random(self._DEBUG_RANDOM_SEED).choice(list(safe_places_dict.keys()), 1)[0]
        new_born_place = safe_places_dict[bp_index]
        new_born_place_config = new_born_place["config"]
        v.vehicle_config.update(new_born_place_config)
        v.reset(self.current_map)
        self._update_destination_for(v)
        v.update_state(detector_mask=None)
        return bp_index


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
                # "show_lidar": True,
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
        d.update({"total_r": total_r, "episode length": ep_s})
        env.render(text=d)
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


def _profile():
    import time
    env = MultiAgentRoundaboutEnv({"num_agents": 40})
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
    # _vis()
    _profile()
    # _long_run()
