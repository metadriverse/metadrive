import gymnasium as gym
import numpy as np

from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.bottleneck import Merge, Split
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.tollgate import TollGate
from metadrive.component.road_network import Road
from metadrive.constants import TerminationState
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.obs.state_obs import LidarStateObservation, StateObservation
from metadrive.utils import Config, clip

MATollConfig = dict(
    spawn_roads=[Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3), -Road(Merge.node(3, 0, 0), Merge.node(3, 0, 1))],
    num_agents=40,
    map_config=dict(exit_length=70, lane_num=3, toll_lane_num=8, toll_length=10),
    top_down_camera_initial_x=125,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=120,
    cross_yellow_line_done=True,
    # ===== Reward Scheme =====
    speed_reward=0.0,
    overspeed_penalty=0.5,
    vehicle_config=dict(
        min_pass_steps=30,  # We ask the agents to stop at tollgate for at least 6s (30 steps).
        show_lidar=False,
        # "show_side_detector": True,
        # "show_lane_line_detector": True,
        side_detector=dict(num_lasers=72, distance=20),  # laser num, distance
        lane_line_detector=dict(num_lasers=4, distance=20),
        lidar=dict(num_lasers=72, distance=20),
    )
)


class StayTimeManager:
    def __init__(self):
        self.entry_time = {}
        self.exit_time = {}
        self.last_block = {}

    def reset(self):
        self.entry_time = {}
        self.exit_time = {}
        self.last_block = {}

    def record(self, agents, time_step):
        for v_id, v in agents.items():
            cur_block_id = v.navigation.current_road.block_ID()
            if v_id in self.last_block:
                last_block_id = self.last_block[v_id]
                self.last_block[v_id] = cur_block_id
                if last_block_id != cur_block_id:
                    if cur_block_id == TollGate.ID:
                        # entry
                        self.entry_time[v_id] = time_step
                    elif (cur_block_id == Merge.ID or cur_block_id == Split.ID) and last_block_id == TollGate.ID:
                        self.exit_time[v_id] = time_step
            else:
                self.last_block[v_id] = cur_block_id


class TollGateStateObservation(StateObservation):
    # no intersection exclude navi info now
    @property
    def observation_space(self):
        # Navi info + Other states
        shape = self.ego_state_obs_dim + self.get_line_detector_dim()
        return gym.spaces.Box(-0.0, 1.0, shape=(shape, ), dtype=np.float32)

    def observe(self, vehicle):
        ego_state = self.vehicle_state(vehicle)
        return ego_state


class TollGateObservation(LidarStateObservation):
    def __init__(self, vehicle_config):
        super(LidarStateObservation, self).__init__(vehicle_config)
        self.state_obs = TollGateStateObservation(vehicle_config)
        self.in_toll_time = 0

    @property
    def observation_space(self):
        shape = list(self.state_obs.observation_space.shape)
        if self.config["vehicle_config"]["lidar"]["num_lasers"] > 0 \
                and self.config["vehicle_config"]["lidar"]["distance"] > 0:
            # Number of lidar rays and distance should be positive!
            shape[0] += self.config["vehicle_config"]["lidar"]["num_lasers"] \
                        + self.config["vehicle_config"]["lidar"]["num_others"] * 4 + 2
        return gym.spaces.Box(-0.0, 1.0, shape=tuple(shape), dtype=np.float32)

    def reset(self, env, vehicle=None):
        self.in_toll_time = 0

    def observe(self, vehicle):
        cur_block_is_toll = vehicle.navigation.current_road.block_ID() == TollGate.ID
        self.in_toll_time += 1 if cur_block_is_toll else 0
        if not cur_block_is_toll:
            toll_obs = [0.0, 0.0]
        else:
            toll_obs = [
                1.0 if cur_block_is_toll else 0.0, 1.0 if self.in_toll_time > vehicle.config["min_pass_steps"] else 0.0
            ]
        # print(toll_obs)
        state = self.state_observe(vehicle)
        other_v_info = self.lidar_observe(vehicle)
        ret = np.concatenate((state, np.asarray(other_v_info), np.asarray(toll_obs)))
        return ret.astype(np.float32)


class MATollGateMap(PGMap):
    BOTTLE_LENGTH = 35

    def _generate(self):
        length = self.config["exit_length"]

        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"

        # Build a first-block
        last_block = FirstPGBlock(
            self.road_network,
            self.config[self.LANE_WIDTH],
            self.config["lane_num"],
            parent_node_path,
            physics_world,
            length=length
        )
        self.blocks.append(last_block)

        split = Split(
            1, last_block.get_socket(index=0), self.road_network, random_seed=1, ignore_intersection_checking=False
        )
        split.construct_block(
            parent_node_path, physics_world, {
                "length": 2,
                "lane_num": self.config["toll_lane_num"] - self.config["lane_num"],
                "bottle_len": self.BOTTLE_LENGTH,
            }
        )
        self.blocks.append(split)
        toll = TollGate(
            2, split.get_socket(index=0), self.road_network, random_seed=1, ignore_intersection_checking=False
        )
        toll.construct_block(parent_node_path, physics_world, {
            "length": self.config["toll_length"],
        })

        self.blocks.append(toll)

        # Build Toll
        merge = Merge(3, toll.get_socket(index=0), self.road_network, random_seed=1, ignore_intersection_checking=False)
        merge.construct_from_config(
            dict(
                lane_num=self.config["toll_lane_num"] - self.config["lane_num"],
                length=self.config["exit_length"],
                bottle_len=self.BOTTLE_LENGTH,
            ), parent_node_path, physics_world
        )
        self.blocks.append(merge)


class MATollGatePGMapManager(PGMapManager):
    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(MATollGateMap, map_config=config["map_config"], random_seed=None)
        else:
            assert len(self.spawned_objects) == 1, "It is supposed to contain one map in this manager"
            _map = self.spawned_objects.values()[0]
        self.load_map(_map)
        self.current_map.spawn_roads = config["spawn_roads"]


class MultiAgentTollgateEnv(MultiAgentMetaDrive):
    def __init__(self, config=None):
        super(MultiAgentTollgateEnv, self).__init__(config)
        self.stay_time_manager = StayTimeManager()

    def reset(self, *args, **kwargs):
        self.stay_time_manager.reset()
        return super(MultiAgentTollgateEnv, self).reset(*args, **kwargs)

    @staticmethod
    def default_config() -> Config:
        assert MATollConfig["vehicle_config"]["side_detector"]["num_lasers"] > 2
        assert MATollConfig["vehicle_config"]["lane_line_detector"]["num_lasers"] > 2
        return MultiAgentMetaDrive.default_config().update(MATollConfig, allow_add_new_key=True)

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
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral_reward"]:
            lateral_factor = clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0

        reward = 0.0
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor

        if vehicle.navigation.current_road.block_ID() == TollGate.ID:
            if vehicle.overspeed:  # Too fast!
                reward = -self.config["overspeed_penalty"] * vehicle.speed_km_h / vehicle.max_speed_km_h
            else:
                # Good! At very low speed
                pass
        else:
            reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h)

        step_info["step_reward"] = reward

        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        return reward, step_info

    def _is_out_of_road(self, vehicle):
        # A specified function to determine whether this vehicle should be done.
        # return vehicle.on_yellow_continuous_line or (not vehicle.on_lane) or vehicle.crash_sidewalk
        ret = vehicle.crash_sidewalk
        if self.config["cross_yellow_line_done"]:
            ret = ret or vehicle.on_yellow_continuous_line
        return ret

    def done_function(self, vehicle_id):
        done, done_info = super(MultiAgentMetaDrive, self).done_function(vehicle_id)
        if done_info["max_step"]:
            return done, done_info

        if done_info[TerminationState.CRASH_VEHICLE] and (not self.config["crash_done"]):
            assert done_info[TerminationState.CRASH_VEHICLE] or done_info[TerminationState.CRASH_BUILDING] or \
                   done_info[TerminationState.SUCCESS] or done_info[TerminationState.OUT_OF_ROAD]
            if not (done_info[TerminationState.SUCCESS] or done_info[TerminationState.OUT_OF_ROAD]):
                # Does not revert done if high-priority termination happens!
                done = False

        if done_info[TerminationState.OUT_OF_ROAD] and (not self.config["out_of_road_done"]):
            assert done_info[TerminationState.CRASH_VEHICLE] or done_info[TerminationState.CRASH_BUILDING] or \
                   done_info[TerminationState.SUCCESS] or done_info[TerminationState.OUT_OF_ROAD]
            if not done_info[TerminationState.SUCCESS]:
                done = False

        if vehicle_id in self.stay_time_manager.entry_time and vehicle_id in self.stay_time_manager.exit_time:
            entry = self.stay_time_manager.entry_time[vehicle_id]
            exit = self.stay_time_manager.exit_time[vehicle_id]
            if (exit - entry) < self.config["vehicle_config"]["min_pass_steps"]:
                done = True
                done_info["out_of_road"] = True

        return done, done_info

    def get_single_observation(self):
        o = TollGateObservation(self.config)
        return o

    def step(self, actions):
        o, r, tm, tc, i = super(MultiAgentTollgateEnv, self).step(actions)
        self.stay_time_manager.record(self.agent_manager.active_agents, self.episode_step)
        return o, r, tm, tc, i

    def setup_engine(self):
        super(MultiAgentTollgateEnv, self).setup_engine()
        self.engine.update_manager("map_manager", MATollGatePGMapManager())


def _draw():
    env = MultiAgentTollgateEnv()
    o, _ = env.reset()
    from metadrive.utils.draw_top_down_map import draw_top_down_map
    import matplotlib.pyplot as plt

    plt.imshow(draw_top_down_map(env.current_map))
    plt.show()
    env.close()


def _expert():
    env = MultiAgentTollgateEnv(
        {
            "vehicle_config": {
                "lidar": {
                    "num_lasers": 240,
                    "num_others": 4,
                    "distance": 50
                },
            },
            "save_level": 1.,
            "use_AI_protector": True,
            "debug_physics_world": True,

            # "use_render": True,
            "debug": True,
            "manual_control": True,
            "num_agents": 4,
        }
    )
    o, _ = env.reset()
    total_r = 0
    ep_s = 0
    for i in range(1, 100000):
        o, r, tm, tc, info = env.step(env.action_space.sample())
        for r_ in r.values():
            total_r += r_
        ep_s += 1
        tm.update({"total_r": total_r, "episode length": ep_s})
        # env.render(text=d)
        if tm["__all__"]:
            print(
                "Finish! Current step {}. Group Reward: {}. Average reward: {}".format(
                    i, total_r, total_r / env.agent_manager.next_agent_count
                )
            )
            break
        if len(env.agents) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()


def _vis_debug_respawn():
    env = MultiAgentTollgateEnv(
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
            "num_agents": 20,
        }
    )
    o, _ = env.reset()
    total_r = 0
    ep_s = 0
    for i in range(1, 100000):
        action = {k: [.0, 1.0] for k in env.agents.keys()}
        o, r, tm, tc, info = env.step(action)
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
        if tm["__all__"]:
            print(
                "Finish! Current step {}. Group Reward: {}. Average reward: {}".format(
                    i, total_r, total_r / env.agent_manager.next_agent_count
                )
            )
            # break
        if len(env.agents) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()


def _vis():
    env = MultiAgentTollgateEnv(
        {
            "horizon": 100000,
            "vehicle_config": {
                "lidar": {
                    "num_lasers": 72,
                    "num_others": 0,
                    "distance": 40
                },
                # "show_lidar": True,
                # "show_side_detector":True,
                # "show_lane_line_detector":True,
            },
            "traffic_density": 0.,
            "traffic_mode": "hybrid",
            "debug": True,
            "use_render": True,
            # "debug": True,
            "manual_control": True,
            "num_agents": 18,
        }
    )
    o, _ = env.reset()
    total_r = 0
    ep_s = 0
    for i in range(1, 100000):
        o, r, tm, tc, info = env.step({k: [0, 1] for k in env.agents.keys()})
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
        track_v = env.agent_manager.object_to_agent(env.current_track_agent.name)
        if track_v in r:
            render_text["tack_v_reward"] = r[track_v]
        render_text["dist_to_right"] = env.current_track_agent.dist_to_right_side
        render_text["dist_to_left"] = env.current_track_agent.dist_to_left_side
        render_text["overspeed"] = env.current_track_agent.overspeed
        render_text["lane"] = env.current_track_agent.lane_index
        render_text["block"] = env.current_track_agent.navigation.current_road.block_ID()
        env.render(text=render_text)
        if tm["__all__"]:
            print(info)
            print(
                "Finish! Current step {}. Group Reward: {}. Average reward: {}".format(
                    i, total_r, total_r / env.agent_manager.next_agent_count
                )
            )
            # break
        if len(env.agents) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()


def _profile():
    import time
    env = MultiAgentTollgateEnv({"num_agents": 8})
    obs, _ = env.reset()
    start = time.time()
    for s in range(10000):
        o, r, tm, tc, i = env.step(env.action_space.sample())

        # mask_ratio = env.engine.detector_mask.get_mask_ratio()
        # print("Mask ratio: ", mask_ratio)

        if all(tm.values()):
            env.reset()
        if (s + 1) % 100 == 0:
            print(
                "Finish {}/10000 simulation steps. Time elapse: {:.4f}. Average FPS: {:.4f}".format(
                    s + 1,
                    time.time() - start, (s + 1) / (time.time() - start)
                )
            )
    print(f"(MetaDriveEnv) Total Time Elapsed: {time.time() - start}")


def _long_run():
    # Please refer to test_ma_Toll_reward_done_alignment()
    _out_of_road_penalty = 3
    env = MultiAgentTollgateEnv(
        {
            "num_agents": 8,
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
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        for step in range(10000):
            act = env.action_space.sample()
            o, r, tm, tc, i = env.step(act)
            if step == 0:
                assert not any(tm.values())

            if any(tm.values()):
                print("Current Done: {}\nReward: {}".format(tm, r))
                for kkk, ddd in tm.items():
                    if ddd and kkk != "__all__":
                        print("Info {}: {}\n".format(kkk, i[kkk]))
                print("\n")

            for kkk, rrr in r.items():
                if rrr == -_out_of_road_penalty:
                    assert tm[kkk]

            if (step + 1) % 200 == 0:
                print(
                    "{}/{} Agents: {} {}\nO: {}\nR: {}\nD: {}\nI: {}\n\n".format(
                        step + 1, 10000, len(env.agents), list(env.agents.keys()),
                        {k: (oo.shape, oo.mean(), oo.min(), oo.max())
                         for k, oo in o.items()}, r, tm, i
                    )
                )
            if tm["__all__"]:
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
    # pygame_replay("tollgate", MultiAgentTollgateEnv)
