import copy
import logging

from pgdrive.envs.marl_envs.marl_inout_roundabout import LidarStateObservationMARound
from pgdrive.envs.marl_envs.multi_agent_pgdrive import MultiAgentPGDrive, panda_replay
from pgdrive.obs.observation_base import ObservationBase
from pgdrive.component.blocks.first_block import FirstPGBlock
from pgdrive.component.blocks.parking_lot import ParkingLot
from pgdrive.component.blocks.t_intersection import TInterSection
from pgdrive.component.map.pg_map import PGMap
from pgdrive.component.road.road import Road
from pgdrive.manager.spawn_manager import SpawnManager
from pgdrive.utils import get_np_random, Config

MAParkingLotConfig = dict(
    num_agents=10,
    crash_done=True,
    parking_space_num=8,
    map_config=dict(exit_length=20, lane_num=1),
    top_down_camera_initial_x=80,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=120,
    vehicle_config={
        "enable_reverse": True,
        "random_navi_mark_color": True,
        "show_dest_mark": True,
        "show_navi_mark": False,
        "show_line_to_dest": True,
    },
)


class ParkingSpaceManager:
    """
    Manage parking spaces, when env.reset() is called, vehicles will be assigned to different spawn points including:
    parking space and entrances of parking lot, vehicle can not respawn in parking space which has been assigned to a
    vehicle who drives into this parking lot.
    """
    def __init__(self, parking_spaces: list):
        self.parking_space_available = set()
        self._parking_spaces = parking_spaces
        self.v_dest_pair = {}
        self.reset()

    def get_parking_space(self, v_id):
        parking_space_idx = get_np_random().choice([i for i in range(len(self.parking_space_available))])
        parking_space = list(self.parking_space_available)[parking_space_idx]
        self.parking_space_available.remove(parking_space)
        self.v_dest_pair[v_id] = parking_space
        return parking_space

    def add_available_parking_space(self, parking_space: Road):
        self.parking_space_available.add(parking_space)

    def after_vehicle_done(self, v_id):
        if v_id in self.v_dest_pair:
            dest = self.v_dest_pair.pop(v_id)
            self.parking_space_available.add(dest)

    def reset(self):
        self.v_dest_pair = {}
        self.parking_space_available = set(copy.deepcopy(self._parking_spaces))


class MAParkingLotMap(PGMap):
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

        last_block = ParkingLot(1, last_block.get_socket(0), self.road_network, 1)
        last_block.construct_block(
            parent_node_path, physics_world, {"one_side_vehicle_number": int(self.config["parking_space_num"] / 2)}
        )
        self.blocks.append(last_block)
        self.parking_space_manager = ParkingSpaceManager(last_block.dest_roads)
        self.parking_lot = last_block

        # Build ParkingLot
        TInterSection.EXIT_PART_LENGTH = 10
        last_block = TInterSection(2, last_block.get_socket(index=0), self.road_network, random_seed=1)
        last_block.construct_block(
            parent_node_path,
            physics_world,
            extra_config={
                "t_type": 1,
                "change_lane_num": 0
                # Note: lane_num is set in config.map_config.lane_num
            }
        )
        self.blocks.append(last_block)


class MultiAgentParkingLotEnv(MultiAgentPGDrive):
    """
    Env will be done when vehicle is on yellow or white continuous lane line!
    """
    in_spawn_roads = [
        Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3),
        -Road(TInterSection.node(2, 0, 0), TInterSection.node(2, 0, 1)),
        -Road(TInterSection.node(2, 2, 0), TInterSection.node(2, 2, 1)),
    ]

    out_spawn_roads = None

    @property
    def spawn_roads(self):
        return self.in_spawn_roads + self.out_spawn_roads

    @staticmethod
    def default_config() -> Config:
        return MultiAgentPGDrive.default_config().update(MAParkingLotConfig, allow_add_new_key=True)

    @staticmethod
    def _get_out_spawn_roads(parking_space_num):
        ret = []
        for i in range(1, parking_space_num + 1):
            ret.append(Road(ParkingLot.node(1, i, 5), ParkingLot.node(1, i, 6)))
        return ret

    def _process_extra_config(self, config) -> "Config":
        ret_config = self.default_config().update(
            config, allow_add_new_key=False, stop_recursive_update=["target_vehicle_configs"]
        )
        if not ret_config["crash_done"] and ret_config["crash_vehicle_penalty"] > 2:
            logging.warning(
                "Are you sure you wish to set crash_vehicle_penalty={} when crash_done=False?".format(
                    ret_config["crash_vehicle_penalty"]
                )
            )
        if ret_config["use_render"] and ret_config["fast"]:
            logging.warning("Turn fast=False can accelerate Multi-agent rendering performance!")

        # add extra assert
        parking_space_num = ret_config["parking_space_num"]
        assert parking_space_num % 2 == 0, "number of parking spaces must be multiples of 2"
        assert parking_space_num >= 4, "minimal number of parking space is 4"
        self.out_spawn_roads = self._get_out_spawn_roads(parking_space_num)
        ret_config["map_config"]["parking_space_num"] = ret_config["parking_space_num"]
        # Workaround
        if ret_config["target_vehicle_configs"]:
            for k, v in ret_config["target_vehicle_configs"].items():
                old = ret_config["vehicle_config"].copy()
                new = old.update(v)
                ret_config["target_vehicle_configs"][k] = new

        self._spawn_manager = SpawnManager(
            exit_length=ret_config["map_config"]["exit_length"],
            lane_num=ret_config["map_config"]["lane_num"],
            num_agents=ret_config["num_agents"],
            vehicle_config=ret_config["vehicle_config"],
            target_vehicle_configs=ret_config["target_vehicle_configs"],
            seed=self._DEBUG_RANDOM_SEED
        )

        self._spawn_manager.set_spawn_roads(self.spawn_roads)

        ret_config = self._update_agent_pos_configs(ret_config)
        return ret_config

    def _update_map(self, episode_data: dict = None, force_seed=None):
        map_config = self.config["map_config"]

        if self.current_map is None:
            self.seed(map_config["seed"])
            new_map = self.engine.map_manager.spawn_object(
                MAParkingLotMap, map_config=map_config, random_seed=self.current_seed
            )
            self.engine.map_manager.load_map(new_map)
            self.current_map.spawn_roads = self.spawn_roads

    def _update_destination_for(self, vehicle_id):
        vehicle = self.vehicles[vehicle_id]
        # when agent re-joined to the game, call this to set the new route to destination
        end_roads = copy.deepcopy(self.in_spawn_roads)
        if vehicle.routing_localization.current_road in end_roads:
            end_road = self.current_map.parking_space_manager.get_parking_space(vehicle_id)
        else:
            end_road = -get_np_random(self._DEBUG_RANDOM_SEED).choice(end_roads)  # Use negative road!
        vehicle.routing_localization.set_route(vehicle.lane_index[0], end_road.end_node)

    def _respawn_single_vehicle(self, randomize_position=False):
        """
        Exclude destination parking space
        """
        safe_places_dict = self._spawn_manager.get_available_respawn_places(
            self.current_map, randomize=randomize_position
        )
        # ===== filter spawn places =====
        filter_ret = {}
        for id, config in safe_places_dict.items():
            spawn_l_index = config["config"]["spawn_lane_index"]
            spawn_road = Road(spawn_l_index[0], spawn_l_index[1])
            if spawn_road in self.in_spawn_roads:
                if len(self.current_map.parking_space_manager.parking_space_available) > 0:
                    filter_ret[id] = config
            else:
                # spawn in parking space
                if ParkingLot.is_in_direction_parking_space(spawn_road):
                    # avoid sweep test bug
                    spawn_road = self.current_map.parking_lot.out_direction_parking_space(spawn_road)
                    config["config"]["spawn_lane_index"] = (spawn_road.start_node, spawn_road.end_node, 0)
                if spawn_road in self.current_map.parking_space_manager.parking_space_available:
                    # not other vehicle's destination
                    filter_ret[id] = config

        # ===== same as super() =====
        safe_places_dict = filter_ret
        if len(safe_places_dict) == 0 or not self.agent_manager.allow_respawn:
            # No more run, just wait!
            return None, None
        assert len(safe_places_dict) > 0
        bp_index = get_np_random(self._DEBUG_RANDOM_SEED).choice(list(safe_places_dict.keys()), 1)[0]
        new_spawn_place = safe_places_dict[bp_index]

        if new_spawn_place[self._spawn_manager.FORCE_AGENT_NAME] is not None:
            if new_spawn_place[self._spawn_manager.FORCE_AGENT_NAME] != self.agent_manager.next_agent_id():
                return None, None

        new_agent_id, vehicle = self.agent_manager.propose_new_vehicle()
        new_spawn_place_config = new_spawn_place["config"]
        vehicle.config.update(new_spawn_place_config)
        vehicle.reset(self.current_map)
        self._update_destination_for(new_agent_id)
        vehicle.after_step(detector_mask=None)
        self.dones[new_agent_id] = False  # Put it in the internal dead-tracking dict.

        new_obs = self.observations[new_agent_id].observe(vehicle)
        return new_agent_id, new_obs

    def get_single_observation(self, vehicle_config: "Config") -> "ObservationBase":
        return LidarStateObservationMARound(vehicle_config)

    def _reset_agents(self):
        self.current_map.parking_space_manager.reset()
        super(MultiAgentParkingLotEnv, self)._reset_agents()

    def done_function(self, vehicle_id):
        done, info = super(MultiAgentParkingLotEnv, self).done_function(vehicle_id)
        if done:
            self.current_map.parking_space_manager.after_vehicle_done(vehicle_id)
        return done, info

    def _is_out_of_road(self, vehicle):
        # A specified function to determine whether this vehicle should be done.
        return vehicle.on_yellow_continuous_line or (not vehicle.on_lane) or vehicle.crash_sidewalk
        # ret = vehicle.out_of_route
        # return ret


def _draw():
    env = MultiAgentParkingLotEnv()
    o = env.reset()
    from pgdrive.utils import draw_top_down_map
    import matplotlib.pyplot as plt

    plt.imshow(draw_top_down_map(env.current_map))
    plt.show()
    env.close()


def _expert():
    env = MultiAgentParkingLotEnv(
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
            "debug_physics_world": True,
            "fast": True,
            # "use_render": True,
            "debug": True,
            "manual_control": True,
            "num_agents": 3,
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
    env = MultiAgentParkingLotEnv(
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
            "fast": True,
            "use_render": True,
            "debug": False,
            "manual_control": True,
            "num_agents": 11,
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
    # vis_big(block_type_version="v2")
    env = MultiAgentParkingLotEnv(
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
            "debug_static_world": True,
            "global_light": True,
            "fast": True,
            "use_render": True,
            "debug": True,
            "manual_control": True,
            "num_agents": 8,
            "delay_done": 10,
            # "parking_space_num": 4
        }
    )
    o = env.reset()
    total_r = 0
    ep_s = 0
    for i in range(1, 100000):
        actions = {k: [1.0, .0] for k in env.vehicles.keys()}
        if len(env.vehicles) == 1:
            actions = {k: [-1.0, .0] for k in env.vehicles.keys()}
        o, r, d, info = env.step(actions)
        for r_ in r.values():
            total_r += r_
        ep_s += 1
        # d.update({"total_r": total_r, "episode length": ep_s})
        if len(env.vehicles) != 0:
            v = env.current_track_vehicle
            dist = v.dist_to_left_side, v.dist_to_right_side
            ckpt_idx = v.routing_localization._target_checkpoints_index
        else:
            dist = (0, 0)
            ckpt_idx = (0, 0)

        render_text = {
            "total_r": total_r,
            "episode length": ep_s,
            "cam_x": env.main_camera.camera_x,
            "cam_y": env.main_camera.camera_y,
            "cam_z": env.main_camera.top_down_camera_height,
            "alive": len(env.vehicles),
            "dist_right_left": dist,
            "ckpt_idx": ckpt_idx,
            "parking_space_num": len(env.current_map.parking_space_manager.parking_space_available)
        }
        if len(env.vehicles) > 0:
            v = env.current_track_vehicle
            # print(v.routing_localization.checkpoints)
            render_text["current_road"] = v.current_road

        env.render(text=render_text)
        for kkk, ddd in d.items():
            if ddd and kkk != "__all__":
                print(
                    "{} done! State: {}".format(
                        kkk, {
                            "arrive_dest": info[kkk]["arrive_dest"],
                            "out_of_road": info[kkk]["out_of_road"],
                            "crash": info[kkk]["crash"],
                            "max_step": info[kkk]["max_step"],
                        }
                    )
                )
        if d["__all__"]:
            print(
                "Finish! Current step {}. Group Reward: {}. Average reward: {}".format(
                    i, total_r, total_r / env.agent_manager.next_agent_count
                )
            )
            env.reset()
            # break
        if len(env.vehicles) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()


def _profile():
    import time
    env = MultiAgentParkingLotEnv({"num_agents": 10})
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
    print(f"(MAParkingLot) Total Time Elapse: {time.time() - start}")


def _long_run():
    # Please refer to test_ma_ParkingLot_reward_done_alignment()
    _out_of_road_penalty = 3
    env = MultiAgentParkingLotEnv(
        {
            "num_agents": 3,
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
    # _vis_debug_respawn()
    _profile()
    # _long_run()
    # pygame_replay("parking", MultiAgentParkingLotEnv, False, other_traj="metasvodist_parking_best.json")
    panda_replay(
        "parking",
        MultiAgentParkingLotEnv,
        False,
        other_traj="metasvodist_parking_best.json",
        extra_config={"global_light": True}
    )
