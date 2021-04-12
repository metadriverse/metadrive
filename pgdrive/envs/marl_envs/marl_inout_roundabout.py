import logging

import numpy as np

from pgdrive.envs.multi_agent_pgdrive import MultiAgentPGDrive
from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.scene_creator.blocks.roundabout import Roundabout
from pgdrive.scene_creator.map import PGMap
from pgdrive.scene_creator.road.road import Road
from pgdrive.utils import get_np_random, PGConfig


class MARoundaboutMap(PGMap):
    def _generate(self, pg_world):
        length = MultiAgentRoundaboutEnv.EXIT_LENGTH

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
                "inner_radius": 40,
                "angle": 70,
                # Note: lane_num is set in config.map_config.lane_num
            }
        )
        self.blocks.append(last_block)


class MultiAgentRoundaboutEnv(MultiAgentPGDrive):
    EXIT_LENGTH = 100
    born_roads = [
        Road(FirstBlock.NODE_2, FirstBlock.NODE_3),
        -Road(Roundabout.node(1, 0, 2), Roundabout.node(1, 0, 3)),
        -Road(Roundabout.node(1, 1, 2), Roundabout.node(1, 1, 3)),
        -Road(Roundabout.node(1, 2, 2), Roundabout.node(1, 2, 3)),
        # -Road(Roundabout.node(1, 3, 2), Roundabout.node(1, 3, 3)),
        # -Road(Roundabout.node(1, 2, 2), Roundabout.node(1, 2, 3)),
    ]

    @staticmethod
    def default_config() -> PGConfig:
        config = MultiAgentPGDrive.default_config()
        config.update(
            {
                "camera_height": 4,
                "map": "M",
                "vehicle_config": {
                    "lidar": {
                        "num_lasers": 120,
                        "distance": 50,
                        "num_others": 4,
                    },
                    "show_lidar": False,
                    "born_longitude": 5,
                    "born_lateral": 0,
                },
                "map_config": {
                    "lane_num": 3
                },
                # clear base config
                "num_agents": 2,
                "auto_termination": False
            },
            allow_overwrite=True,
        )
        return config

    def _update_map(self, episode_data: dict = None):
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
            bird_camera_height = 240
            self.main_camera.camera.setPos(0, 0, bird_camera_height)
            self.main_camera.bird_camera_height = bird_camera_height
            self.main_camera.stop_chase(self.pg_world)
            # self.main_camera.camera.setPos(300, 20, bird_camera_height)
            self.main_camera.camera_x += 140
            self.main_camera.camera_y += 20

    def _process_extra_config(self, config):
        config = super(MultiAgentRoundaboutEnv, self)._process_extra_config(config)
        config = self._update_agent_pos_configs(config)
        return super(MultiAgentRoundaboutEnv, self)._process_extra_config(config)

    def _update_agent_pos_configs(self, config):
        target_vehicle_configs = []
        self._all_lane_index = []
        self._next_agent_id = config["num_agents"]

        num_concurrent = 3
        assert config["num_agents"] <= config["map_config"]["lane_num"] * len(self.born_roads) * num_concurrent, (
            "Too many agents! We only accepet {} agents, but you have {} agents!".format(
                config["map_config"]["lane_num"] * len(self.born_roads) * num_concurrent, config["num_agents"]
            )
        )

        # We can spawn agents in the middle of road at the initial time, but when some vehicles need to be reborn,
        # then we have to set it to the farthest places to ensure safety (otherwise the new vehicles may suddenly
        # appear at the middle of the road!)
        self._safe_born_places = []
        self._last_born_identifier = None
        for i, road in enumerate(self.born_roads):
            for lane_idx in range(config["map_config"]["lane_num"]):
                for j in range(num_concurrent):
                    interval = self.EXIT_LENGTH / num_concurrent
                    long = j * interval + np.random.uniform(0, 0.5 * interval)
                    target_vehicle_configs.append(
                        (
                            "agent_{}_{}".format(i + 1, lane_idx), {
                                "born_lane_index": road.lane_index(lane_idx),
                                "born_longitude": long
                            }
                        )
                    )
                    self._all_lane_index.append(road.lane_index(lane_idx))
                    if j == 0:
                        self._safe_born_places.append(
                            dict(
                                identifier=road.lane_index(lane_idx)[0],  # identifier
                                config={
                                    "born_lane_index": road.lane_index(lane_idx),
                                    "born_longitude": long
                                }
                            )
                        )

        target_agents = get_np_random().choice(
            [i for i in range(len(target_vehicle_configs))], config["num_agents"], replace=False
        )

        # for rllib compatibility
        ret = {}
        if len(target_agents) > 1:
            for real_idx, idx in enumerate(target_agents):
                agent_name, v_config = target_vehicle_configs[idx]
                ret["agent{}".format(real_idx)] = v_config
        else:
            agent_name, v_config = target_vehicle_configs[0]
            ret[self.DEFAULT_AGENT] = dict(born_lane_index=v_config)
        config["target_vehicle_configs"] = ret
        return config

    def reset(self, episode_data: dict = None):
        self._next_agent_id = self.num_agents
        self._last_born_identifier = 0
        ret = super(MultiAgentRoundaboutEnv, self).reset(episode_data)
        self.for_each_vehicle(self._update_destination_for)
        return ret

    def step(self, actions):
        o, r, d, i = super(MultiAgentRoundaboutEnv, self).step(actions)
        if self.num_agents > 1:
            d["__all__"] = False  # Never done
        return o, r, d, i

    def _update_destination_for(self, vehicle):
        # when agent re-joined to the game, call this to set the new route to destination
        end_road = -get_np_random().choice(self.born_roads)  # Use negative road!
        vehicle.routing_localization.set_route(vehicle.lane_index[0], end_road.end_node)

    def _reborn(self, dead_vehicle_id):
        assert dead_vehicle_id in self.vehicles
        # Switch to track other vehicle if in first-person view.
        # if self.config["use_render"] and self.current_track_vehicle_id == id:
        #     self.chase_another_v()

        v = self.vehicles.pop(dead_vehicle_id)
        v.prepare_step([0, -1])

        # register vehicle
        new_id = "agent{}".format(self._next_agent_id)
        self._next_agent_id += 1
        self.vehicles[new_id] = v  # Put it to new vehicle id.
        self.dones[new_id] = False  # Put it in the internal dead-tracking dict.
        logging.debug("{} Dead. {} Reborn!".format(dead_vehicle_id, new_id))

        # replace vehicle to new born place
        safe_places = [p for p in self._safe_born_places if p['identifier'] != self._last_born_identifier]
        new_born_place = safe_places[get_np_random().choice(len(safe_places), 1)[0]]
        new_born_place_config = new_born_place["config"]
        self._last_born_identifier = new_born_place["identifier"]
        v.vehicle_config.update(new_born_place_config)
        v.reset(self.current_map)

        # reset observation space
        obs = self.observations.pop(dead_vehicle_id)
        self.observations[new_id] = obs
        self.observations[new_id].reset(self, v)
        new_obs = self.observations[new_id].observe(v)
        self.observation_space = self._get_observation_space()

        # reset action space
        self.action_space = self._get_action_space()
        return new_obs, new_id

    def _after_vehicle_done(self, obs=None, reward=None, dones: dict = None, info=None):
        dones = self._wrap_as_multi_agent(dones)
        new_dones = dict()
        for dead_vehicle_id, done in dones.items():
            new_dones[dead_vehicle_id] = done
            if done:
                new_obs, new_id = self._reborn(dead_vehicle_id)
                obs[new_id] = new_obs
                reward[new_id] = 0.0
                info[new_id] = {}
                new_dones[new_id] = False
        return obs, reward, new_dones, info


def _draw():
    env = MultiAgentRoundaboutEnv()
    o = env.reset()
    from pgdrive.utils import draw_top_down_map
    import matplotlib.pyplot as plt

    plt.imshow(draw_top_down_map(env.current_map))
    plt.show()
    env.close()


def _vis():
    env = MultiAgentRoundaboutEnv(
        {
            "vehicle_config": {
                "lidar": {
                    "num_lasers": 120,
                    "distance": 50
                }
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
        o, r, d, info = env.step(env.action_space.sample())
        if env.num_agents == 1:
            r = env._wrap_as_multi_agent(r)
        for r_ in r.values():
            total_r += r_
        o, r, d, info = env.step(env.action_space.sample())
        ep_s += 1
        d.update({"total_r": total_r, "episode length": ep_s})
        env.render(text=d)
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


if __name__ == "__main__":
    # _draw()
    # _vis()
    _profile()
