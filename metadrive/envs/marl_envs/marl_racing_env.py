
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.component.map.pg_map import PGMap
from metadrive.component.pg_space import Parameter
from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.roundabout import Roundabout
from metadrive.component.pgblock.straight import Straight
from metadrive.component.road_network import Road
from metadrive.manager.pg_map_manager import PGMapManager
import copy
from metadrive import MetaDriveEnv
from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.roundabout import Roundabout
from metadrive.component.road_network import Road
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.manager.spawn_manager import SpawnManager
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.utils import Config


import argparse
import cv2
import numpy as np

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE



from metadrive.component.map.pg_map import PGMap
from metadrive.component.pg_space import Parameter
from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.roundabout import Roundabout
from metadrive.component.pgblock.straight import Straight
from metadrive.component.road_network import Road
from metadrive.manager.pg_map_manager import PGMapManager
import copy
from metadrive import MetaDriveEnv
from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.roundabout import Roundabout
from metadrive.component.road_network import Road
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.manager.spawn_manager import SpawnManager
from metadrive.utils import Config


import argparse
import cv2
import numpy as np

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE
from metadrive.utils import Config
from metadrive.utils.math import clip

Racing_config = dict(
    # controller="joystick",
    num_agents=2,
    use_render=False,
    manual_control=True,
    traffic_density=0,
    num_scenarios=10000,
    random_agent_model=False,
    top_down_camera_initial_x=95,
    top_down_camera_initial_y=15,
    top_down_camera_initial_z=120,
    # random_lane_width=True,
    # random_lane_num=True,
    cross_yellow_line_done=True,
    use_lateral = False,

    on_continuous_line_done=False,
    out_of_route_done=True,
    vehicle_config=dict(show_lidar=False, show_navi_mark=False),
)


class MultiAgentRacingMap(PGMap):
    def _generate(self):
        FirstPGBlock.ENTRANCE_LENGTH = 0.5
        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"

        # test = TestBlock(False)
        # initialize_asset_loader(engine=test)
        # global_network = NodeRoadNetwork()
        blocks = []
        init_block = FirstPGBlock(self.road_network, 3.0, 3, parent_node_path, physics_world, 1)
        self.blocks.append(init_block)

        # block_s1 = Straight(1, init_block.get_socket(0), self.road_network, 1)
        # block_s1.construct_from_config(
        #     {
        #         Parameter.length: 100
        #     }, parent_node_path, physics_world
        # )
        # self.blocks.append(block_s1)

        block_s1 = Straight(1, init_block.get_socket(0),  self.road_network, 1)
        block_s1.construct_from_config(
            {
                Parameter.length: 100
            },parent_node_path, physics_world
        )
        self.blocks.append(block_s1)

        block_c1 = Curve(2, block_s1.get_socket(0),  self.road_network, 1)
        block_c1.construct_from_config({
            Parameter.length: 200,
            Parameter.radius: 100,
            Parameter.angle: 90,
            Parameter.dir: 1,
        }, parent_node_path, physics_world)
        self.blocks.append(block_c1)

        block_s2 = Straight(3, block_c1.get_socket(0),  self.road_network, 1)
        block_s2.construct_from_config(
            {
                Parameter.length: 100,
            },parent_node_path, physics_world
        )
        self.blocks.append(block_s2)

        block_c2 = Curve(4, block_s2.get_socket(0),  self.road_network, 1)
        block_c2.construct_from_config({
            Parameter.length: 100,
            Parameter.radius: 60,
            Parameter.angle: 90,
            Parameter.dir: 1,
        }, parent_node_path, physics_world)
        self.blocks.append(block_c2)

        block_c3 = Curve(5, block_c2.get_socket(0),  self.road_network, 1)
        block_c3.construct_from_config({
            Parameter.length: 100,
            Parameter.radius: 60,
            Parameter.angle: 90,
            Parameter.dir: 1,
        }, parent_node_path, physics_world)
        self.blocks.append(block_c3)

        block_s3 = Straight(6, block_c3.get_socket(0),  self.road_network, 1)
        block_s3.construct_from_config(
            {
                Parameter.length: 200,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_s3)

        block_c4 = Curve(7, block_s3.get_socket(0),  self.road_network, 1)
        block_c4.construct_from_config({
            Parameter.length: 80,
            Parameter.radius: 40,
            Parameter.angle: 90,
            Parameter.dir: 1,
        }, parent_node_path, physics_world)
        self.blocks.append(block_c4)

        block_c5 = Curve(8, block_c4.get_socket(0),  self.road_network, 1)
        block_c5.construct_from_config({
            Parameter.length: 40,
            Parameter.radius: 50,
            Parameter.angle: 180,
            Parameter.dir: 1,
        }, parent_node_path, physics_world)
        self.blocks.append(block_c5)

        block_c6 = Curve(9, block_c5.get_socket(0), self.road_network, 1)
        block_c6.construct_from_config({
            Parameter.length: 40,
            Parameter.radius: 50,
            Parameter.angle: 220,
            Parameter.dir: 0,
        }, parent_node_path, physics_world)
        self.blocks.append(block_c6)

        block_c7 = Curve(10, block_c6.get_socket(0), self.road_network, 1)
        block_c7.construct_from_config({
            Parameter.length: 40,
            Parameter.radius: 20,
            Parameter.angle: 180,
            Parameter.dir: 1,
        }, parent_node_path, physics_world)
        self.blocks.append(block_c7)

        block_s4 = Straight(11, block_c7.get_socket(0), self.road_network, 1)
        block_s4.construct_from_config({
            Parameter.length: 100,
        }, parent_node_path, physics_world)
        self.blocks.append(block_s4)

        block_c8 = Curve(12, block_s4.get_socket(0), self.road_network, 1)
        block_c8.construct_from_config({
            Parameter.length: 100,
            Parameter.radius: 40,
            Parameter.angle: 140,
            Parameter.dir: 0,
        }, parent_node_path, physics_world)
        self.blocks.append(block_c8)

        # # to connect last road and the first
        # pos_from_node = list(self.road_network.graph.keys())[-2]
        # pos_to_node = list(self.road_network.graph[pos_from_node].keys())[-1]
        # self.road_network.graph[pos_to_node] = {
        #     '>': self.road_network.graph[pos_from_node][pos_to_node]
        # }





# class RoundaboutSpawnManager(SpawnManager):
#     def update_destination_for(self, vehicle_id, vehicle_config):
#         end_roads = copy.deepcopy(self.engine.global_config["spawn_roads"])
#         end_road = -self.np_random.choice(end_roads)  # Use negative road!
#         vehicle_config["destination"] = end_road.end_node
#         return vehicle_config
#



class MultiAgentRacingMapManager(PGMapManager):
    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(MultiAgentRacingMap, map_config=config["map_config"], random_seed=None)
        else:
            assert len(self.spawned_objects) == 1, "It is supposed to contain one map in this manager"
            _map = self.spawned_objects.values()[0]
        self.load_map(_map)
        # self.current_map.spawn_roads = config["spawn_roads"]





class MultiAgentRacingEnv(MultiAgentMetaDrive):

    @staticmethod
    def default_config() -> Config:
        # assert MABidirectionConfig["vehicle_config"]["side_detector"]["num_lasers"] > 2
        # assert MABidirectionConfig["vehicle_config"]["lane_line_detector"]["num_lasers"] > 2
        # MABidirectionConfig["map_config"]["lane_num"] = MABidirectionConfig["map_config"]["bottle_lane_num"]
        return MultiAgentMetaDrive.default_config().update(Racing_config, allow_add_new_key=True)


    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        vehicle = self.vehicles[vehicle_id]
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
        if self.config["use_lateral"]:
            lateral_factor = clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0
        lateral_factor = 1.0

        reward = 0.0
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor
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
        ret = vehicle.on_white_continuous_line or (not vehicle.on_lane) or vehicle.crash_sidewalk
        if self.config["cross_yellow_line_done"]:
            ret = ret or vehicle.on_yellow_continuous_line
        return ret

    def setup_engine(self):
        super(MultiAgentRacingEnv, self).setup_engine()
        # self.engine.update_manager("spawn_manager", RoundaboutSpawnManager())
        self.engine.update_manager("map_manager", MultiAgentRacingMapManager())




def _draw():
    env = MultiAgentRacingEnv()
    o, _ = env.reset()
    from metadrive.utils.draw_top_down_map import draw_top_down_map
    import matplotlib.pyplot as plt

    plt.imshow(draw_top_down_map(env.current_map))
    plt.show()
    env.close()


def _expert():
    env = MultiAgentRacingEnv(
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

            "use_render": False,
            "debug": True,
            "manual_control": True,
            "num_agents": 2,
            "agent_policy": IDMPolicy,
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
        if len(env.vehicles) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()


def _vis_debug_respawn():
    env = MultiAgentRacingEnv(
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
            "use_render": False,
            "debug": False,
            "manual_control": True,
            "num_agents": 2,
        }
    )
    o, _ = env.reset()
    total_r = 0
    ep_s = 0
    for i in range(1, 100000):
        action = {k: [.0, 1.0] for k in env.vehicles.keys()}
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
        if len(env.vehicles) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()


def _vis():
    env = MultiAgentRacingEnv(
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
            "use_render": False,
            # "debug": True,
            "manual_control": True,
            "num_agents": 2,
        }
    )
    o, _ = env.reset()
    total_r = 0
    ep_s = 0
    for i in range(1, 100000):
        o, r, tm, tc, info = env.step({k: [1.0, .0] for k in env.vehicles.keys()})
        for r_ in r.values():
            total_r += r_
        ep_s += 1
        # d.update({"total_r": total_r, "episode length": ep_s})
        render_text = {
            "total_r": total_r,
            "episode length": ep_s,
            "cam_x": env.main_camera.camera_x,
            "cam_y": env.main_camera.camera_y,
            "cam_z": env.main_camera.top_down_camera_height,
            "current_track_v": env.agent_manager.object_to_agent(env.current_track_vehicle.name)
        }
        track_v = env.agent_manager.object_to_agent(env.current_track_vehicle.name)
        render_text["tack_v_reward"] = r[track_v]
        render_text["dist_to_right"] = env.current_track_vehicle.dist_to_right_side
        render_text["dist_to_left"] = env.current_track_vehicle.dist_to_left_side
        env.render(text=render_text)
        if tm["__all__"]:
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
    env = MultiAgentRacingEnv({"num_agents": 2})
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
    print(f"(MetaDriveEnv) Total Time Elapse: {time.time() - start}")


def _long_run():
    # Please refer to test_ma_Bidirection_reward_done_alignment()
    _out_of_road_penalty = 3
    env = MultiAgentRacingEnv(
        {
            "num_agents": 2,
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
                print("Current Done: {}\nReward: {}".format(d, r))
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
                        step + 1, 10000, len(env.vehicles), list(env.vehicles.keys()),
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
    # pygame_replay("bottle")








