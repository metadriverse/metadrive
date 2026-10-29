import copy

import numpy as np
from panda3d.bullet import BulletBoxShape, BulletGhostNode
from panda3d.core import Vec3

from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.parking_lot import ParkingLot
from metadrive.component.pgblock.t_intersection import TInterSection
from metadrive.component.road_network import Road
from metadrive.constants import MetaDriveType
from metadrive.constants import CollisionGroup
from metadrive.engine.engine_utils import get_engine
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.utils import get_np_random, Config
from metadrive.utils.coordinates_shift import panda_vector, panda_heading
from metadrive.utils.pg.utils import rect_region_detection

MAParkingLotConfig = dict(
    in_spawn_roads=[
        Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3),
        -Road(TInterSection.node(2, 0, 0), TInterSection.node(2, 0, 1)),
        -Road(TInterSection.node(2, 2, 0), TInterSection.node(2, 2, 1)),
    ],
    out_spawn_roads=None,  # auto fill
    spawn_roads=None,  # auto fill
    num_agents=10,
    parking_space_num=8,
    map_config=dict(exit_length=20, lane_num=1),
    top_down_camera_initial_x=80,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=120,
    vehicle_config={
        "enable_reverse": True,
        "show_dest_mark": True,
        "show_navi_mark": False,
        "show_line_to_dest": True,
    },
)

from metadrive.manager.spawn_manager import SpawnManager


class ParkingLotSpawnManager(SpawnManager):
    """
    Manage parking spaces, when env.reset() is called, vehicles will be assigned to different spawn points including:
    parking space and entrances of parking lot, vehicle can not respawn in parking space which has been assigned to a
    vehicle who drives into this parking lot.
    """
    def __init__(self):
        super(ParkingLotSpawnManager, self).__init__()
        self.parking_space_available = set()
        self._parking_spaces = None
        self.v_dest_pair = {}

    def get_parking_space(self, v_id):
        if self._parking_spaces is None:
            self._parking_spaces = self.engine.map_manager.current_map.parking_space
            self.v_dest_pair = {}
            self.parking_space_available = set(copy.deepcopy(self._parking_spaces))
        assert len(self.parking_space_available) > 0
        parking_space_idx = self.np_random.choice([i for i in range(len(self.parking_space_available))])
        parking_space = list(self.parking_space_available)[parking_space_idx]
        self.parking_space_available.remove(parking_space)
        self.v_dest_pair[v_id] = parking_space
        return parking_space

    def after_vehicle_done(self, v_id):
        if v_id in self.v_dest_pair:
            dest = self.v_dest_pair.pop(v_id)
            self.parking_space_available.add(dest)

    def reset(self):
        self._parking_spaces = self.engine.map_manager.current_map.parking_space
        self.v_dest_pair = {}
        self.parking_space_available = set(copy.deepcopy(self._parking_spaces))
        super(ParkingLotSpawnManager, self).reset()

    def update_destination_for(self, vehicle_id, vehicle_config):
        # when agent re-joined to the game, call this to set the new route to destination
        end_roads = copy.deepcopy(self.engine.global_config["in_spawn_roads"])
        if Road(*vehicle_config["spawn_lane_index"][:-1]) in end_roads:
            end_road = self.engine.spawn_manager.get_parking_space(vehicle_id)
        else:
            end_road = -self.np_random.choice(end_roads)  # Use negative road!
        vehicle_config["destination"] = end_road.end_node
        return vehicle_config

    def get_available_respawn_places(self, map, randomize=False):
        """
        In each episode, we allow the vehicles to respawn at the start of road, randomize will give vehicles a random
        position in the respawn region
        """
        engine = get_engine()
        ret = {}
        for bid, bp in self.safe_spawn_places.items():
            if bid in self.spawn_places_used:
                continue
            if "P" not in bid and len(self.parking_space_available) == 0:
                # If no parking space, vehicles will never be spawned.
                continue
            # save time calculate once
            if not bp.get("spawn_point_position", False):
                lane = map.road_network.get_lane(bp["config"]["spawn_lane_index"])
                assert isinstance(lane, StraightLane), "Now we don't support respawn on circular lane"
                long = self.RESPAWN_REGION_LONGITUDE / 2
                spawn_point_position = lane.position(longitudinal=long, lateral=0)
                bp.force_update(
                    {
                        "spawn_point_heading": np.rad2deg(lane.heading_theta_at(long)),
                        "spawn_point_position": (spawn_point_position[0], spawn_point_position[1])
                    }
                )

            spawn_point_position = bp["spawn_point_position"]
            lane_heading = bp["spawn_point_heading"]
            result = rect_region_detection(
                engine, spawn_point_position, lane_heading, self.RESPAWN_REGION_LONGITUDE, self.RESPAWN_REGION_LATERAL,
                CollisionGroup.Vehicle
            )
            if (engine.global_config["debug"] or engine.global_config["debug_physics_world"]) \
                    and bp.get("need_debug", True):
                shape = BulletBoxShape(Vec3(self.RESPAWN_REGION_LONGITUDE / 2, self.RESPAWN_REGION_LATERAL / 2, 1))
                vis_body = engine.render.attach_new_node(BulletGhostNode("debug"))
                vis_body.node().addShape(shape)
                vis_body.setH(panda_heading(lane_heading))
                vis_body.setPos(panda_vector(spawn_point_position, z=2))
                engine.physics_world.dynamic_world.attach(vis_body.node())
                vis_body.node().setIntoCollideMask(CollisionGroup.AllOff)
                bp.force_set("need_debug", False)

            if not result.hasHit() or result.node.getName() != MetaDriveType.VEHICLE:
                new_bp = copy.deepcopy(bp).get_dict()
                if randomize:
                    new_bp["config"] = self._randomize_position_in_slot(new_bp["config"])
                ret[bid] = new_bp
                self.spawn_places_used.append(bid)
        return ret


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

        last_block = ParkingLot(1, last_block.get_socket(0), self.road_network, 1, ignore_intersection_checking=False)
        last_block.construct_block(
            parent_node_path, physics_world, {"one_side_vehicle_number": int(self.config["parking_space_num"] / 2)}
        )
        self.blocks.append(last_block)
        self.parking_space = last_block.dest_roads
        self.parking_lot = last_block

        # Build ParkingLot
        TInterSection.EXIT_PART_LENGTH = 10
        last_block = TInterSection(
            2, last_block.get_socket(index=0), self.road_network, random_seed=1, ignore_intersection_checking=False
        )
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


class MAParkinglotPGMapManager(PGMapManager):
    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(MAParkingLotMap, map_config=config["map_config"], random_seed=None)
        else:
            assert len(self.spawned_objects) == 1, "It is supposed to contain one map in this manager"
            _map = self.spawned_objects.values()[0]
        self.load_map(_map)
        self.current_map.spawn_roads = config["spawn_roads"]


class MultiAgentParkingLotEnv(MultiAgentMetaDrive):
    """
    Env will be done when vehicle is on yellow or white continuous lane line!
    """
    @staticmethod
    def default_config() -> Config:
        return MultiAgentMetaDrive.default_config().update(MAParkingLotConfig, allow_add_new_key=True)

    @staticmethod
    def _get_out_spawn_roads(parking_space_num):
        ret = []
        for i in range(1, parking_space_num + 1):
            ret.append(Road(ParkingLot.node(1, i, 5), ParkingLot.node(1, i, 6)))
        return ret

    def _post_process_config(self, config):
        ret_config = super(MultiAgentParkingLotEnv, self)._post_process_config(config)
        # add extra assert
        parking_space_num = ret_config["parking_space_num"]
        assert parking_space_num % 2 == 0, "number of parking spaces must be multiples of 2"
        assert parking_space_num >= 4, "minimal number of parking space is 4"
        ret_config["out_spawn_roads"] = self._get_out_spawn_roads(parking_space_num)
        ret_config["spawn_roads"] = ret_config["in_spawn_roads"] + ret_config["out_spawn_roads"]
        ret_config["map_config"]["parking_space_num"] = ret_config["parking_space_num"]
        return ret_config

    def _respawn_single_vehicle(self, randomize_position=False):
        """
        Exclude destination parking space
        """
        safe_places_dict = self.engine.spawn_manager.get_available_respawn_places(
            self.current_map, randomize=randomize_position
        )
        # ===== filter spawn places =====
        filter_ret = {}
        for id, config in safe_places_dict.items():
            spawn_l_index = config["config"]["spawn_lane_index"]
            spawn_road = Road(spawn_l_index[0], spawn_l_index[1])
            if spawn_road in self.config["in_spawn_roads"]:
                if len(self.engine.spawn_manager.parking_space_available) > 0:
                    filter_ret[id] = config
            else:
                # spawn in parking space
                if ParkingLot.is_in_direction_parking_space(spawn_road):
                    # avoid sweep test bug
                    spawn_road = self.current_map.parking_lot.out_direction_parking_space(spawn_road)
                    config["config"]["spawn_lane_index"] = (spawn_road.start_node, spawn_road.end_node, 0)
                if spawn_road in self.engine.spawn_manager.parking_space_available:
                    # not other vehicle's destination
                    filter_ret[id] = config

        # ===== same as super() =====
        if len(safe_places_dict) == 0:
            return None, None, None
        born_place_index = get_np_random(self._DEBUG_RANDOM_SEED).choice(list(safe_places_dict.keys()), 1)[0]
        new_spawn_place = safe_places_dict[born_place_index]

        new_agent_id, vehicle, step_info = self.agent_manager.propose_new_vehicle()
        new_spawn_place_config = new_spawn_place["config"]
        new_spawn_place_config = self.engine.spawn_manager.update_destination_for(new_agent_id, new_spawn_place_config)
        vehicle.config.update(new_spawn_place_config)
        vehicle.reset()
        after_step_info = vehicle.after_step()
        step_info.update(after_step_info)
        self.dones[new_agent_id] = False  # Put it in the internal dead-tracking dict.

        new_obs = self.observations[new_agent_id].observe(vehicle)
        return new_agent_id, new_obs, step_info

    def done_function(self, vehicle_id):
        done, info = super(MultiAgentParkingLotEnv, self).done_function(vehicle_id)
        if done:
            self.engine.spawn_manager.after_vehicle_done(vehicle_id)
        return done, info

    def _is_out_of_road(self, vehicle):
        # A specified function to determine whether this vehicle should be done.
        return vehicle.on_yellow_continuous_line or (not vehicle.on_lane) or vehicle.crash_sidewalk
        # ret = vehicle.out_of_route
        # return ret

    def setup_engine(self):
        super(MultiAgentParkingLotEnv, self).setup_engine()
        self.engine.update_manager("spawn_manager", ParkingLotSpawnManager())
        self.engine.update_manager("map_manager", MAParkinglotPGMapManager())


def _draw():
    env = MultiAgentParkingLotEnv()
    o, _ = env.reset()
    from metadrive.utils.draw_top_down_map import draw_top_down_map
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
            },
            "save_level": 1.,
            "use_AI_protector": True,
            "debug_physics_world": True,

            # "use_render": True,
            "debug": True,
            "manual_control": True,
            "num_agents": 3,
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
            "use_render": True,
            "debug": False,
            "manual_control": True,
            "num_agents": 11,
        }
    )
    o, _ = env.reset()
    total_r = 0
    ep_s = 0
    for i in range(1, 100000):
        action = {k: [0.0, .0] for k in env.agents.keys()}
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
            "debug_static_world": False,
            "debug_physics_world": False,
            "use_render": True,
            "debug": True,
            "manual_control": True,
            "num_agents": 7,
            "delay_done": 10,
            # "parking_space_num": 4
        }
    )
    o, _ = env.reset()
    total_r = 0
    ep_s = 0
    for i in range(1, 100000):
        actions = {k: [1.0, .0] for k in env.agents.keys()}
        if len(env.agents) == 1:
            actions = {k: [-1.0, .0] for k in env.agents.keys()}
        o, r, tm, tc, info = env.step(actions)
        for r_ in r.values():
            total_r += r_
        ep_s += 1
        # d.update({"total_r": total_r, "episode length": ep_s})
        if len(env.agents) != 0:
            v = env.current_track_agent
            dist = v.dist_to_left_side, v.dist_to_right_side
            ckpt_idx = v.navigation._target_checkpoints_index
        else:
            dist = (0, 0)
            ckpt_idx = (0, 0)

        render_text = {
            "total_r": total_r,
            "episode length": ep_s,
            "cam_x": env.main_camera.camera_x,
            "cam_y": env.main_camera.camera_y,
            "cam_z": env.main_camera.top_down_camera_height,
            "alive": len(env.agents),
            "dist_right_left": dist,
            "ckpt_idx": ckpt_idx,
            "parking_space_num": len(env.engine.spawn_manager.parking_space_available)
        }
        if len(env.agents) > 0:
            v = env.current_track_agent
            # print(v.navigation.checkpoints)
            render_text["current_road"] = v.navigation.current_road

        env.render(text=render_text)
        d = tm
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
        if tm["__all__"]:
            print(
                "Finish! Current step {}. Group Reward: {}. Average reward: {}".format(
                    i, total_r, total_r / env.agent_manager.next_agent_count
                )
            )
            env.reset()
            # break
        if len(env.agents) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()


def _profile():
    import time
    env = MultiAgentParkingLotEnv({"num_agents": 10})
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
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        for step in range(10000):
            act = env.action_space.sample()
            o, r, tm, tc, i = env.step(act)
            d = tm
            if step == 0:
                assert not any(d.values())

            if any(tm.values()):
                print("Current Done: {}\nReward: {}".format(d, r))
                for kkk, ddd in tm.items():
                    if ddd and kkk != "__all__":
                        print("Info {}: {}\n".format(kkk, i[kkk]))
                print("\n")

            for kkk, rrr in r.items():
                if rrr == -_out_of_road_penalty:
                    assert d[kkk]

            if (step + 1) % 200 == 0:
                print(
                    "{}/{} Agents: {} {}\nO: {}\nR: {}\nD: {}\nI: {}\n\n".format(
                        step + 1, 10000, len(env.agents), list(env.agents.keys()),
                        {k: (oo.shape, oo.mean(), oo.min(), oo.max())
                         for k, oo in o.items()}, r, d, i
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
    # pygame_replay("parking", MultiAgentParkingLotEnv, False, other_traj="metasvodist_parking_best.json")
    # panda_replay(
    #     "parking",
    #     MultiAgentParkingLotEnv,
    #     False,
    #     other_traj="metasvodist_parking_best.json",
    # )
