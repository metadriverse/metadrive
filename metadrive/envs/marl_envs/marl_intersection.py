import copy

from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.intersection import InterSection
from metadrive.component.road_network import Road
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.map_manager import PGMapManager
from metadrive.manager.spawn_manager import SpawnManager
from metadrive.utils import Config

MAIntersectionConfig = dict(
    spawn_roads=[
        Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3),
        -Road(InterSection.node(1, 0, 0), InterSection.node(1, 0, 1)),
        -Road(InterSection.node(1, 1, 0), InterSection.node(1, 1, 1)),
        -Road(InterSection.node(1, 2, 0), InterSection.node(1, 2, 1)),
    ],
    num_agents=30,
    map_config=dict(exit_length=60, lane_num=2),
    top_down_camera_initial_x=80,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=120
)


class MAIntersectionMap(PGMap):
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

        # Build Intersection
        InterSection.EXIT_PART_LENGTH = length

        if "radius" in self.config and self.config["radius"]:
            extra_kwargs = dict(radius=self.config["radius"])
        else:
            extra_kwargs = {}
        last_block = InterSection(
            1,
            last_block.get_socket(index=0),
            self.road_network,
            random_seed=1,
            ignore_intersection_checking=False,
            **extra_kwargs
        )

        if self.config["lane_num"] > 1:
            # We disable U turn in TinyInter environment!
            last_block.enable_u_turn(True)
        else:
            last_block.enable_u_turn(False)

        last_block.construct_block(parent_node_path, physics_world)
        self.blocks.append(last_block)


class MAIntersectionSpawnManager(SpawnManager):
    def __init__(self, disable_u_turn=False):
        super(MAIntersectionSpawnManager, self).__init__()
        self.disable_u_turn = disable_u_turn

    def update_destination_for(self, agent_id, vehicle_config):
        end_roads = copy.deepcopy(self.engine.global_config["spawn_roads"])
        if self.disable_u_turn:  # Remove the spawn road from end roads
            end_roads = [r for r in end_roads if Road(*vehicle_config["spawn_lane_index"][:2]) != r]
        end_road = -self.np_random.choice(end_roads)  # Use negative road!
        vehicle_config["destination"] = end_road.end_node
        return vehicle_config


class MAIntersectionPGMapManager(PGMapManager):
    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(MAIntersectionMap, map_config=config["map_config"], random_seed=None)
        else:
            assert len(self.spawned_objects) == 1, "It is supposed to contain one map in this manager"
            _map = self.spawned_objects.values()[0]
        self.load_map(_map)
        self.current_map.spawn_roads = config["spawn_roads"]


class MultiAgentIntersectionEnv(MultiAgentMetaDrive):
    @staticmethod
    def default_config() -> Config:
        return MultiAgentMetaDrive.default_config().update(MAIntersectionConfig, allow_add_new_key=True)

    def setup_engine(self):
        disable_u_turn = self.config["map_config"]["lane_num"] < 2
        super(MultiAgentIntersectionEnv, self).setup_engine()
        self.engine.update_manager("map_manager", MAIntersectionPGMapManager())
        self.engine.update_manager("spawn_manager", MAIntersectionSpawnManager(disable_u_turn=disable_u_turn))


def _draw():
    env = MultiAgentIntersectionEnv()
    o = env.reset()
    from metadrive.utils.draw_top_down_map import draw_top_down_map
    import matplotlib.pyplot as plt

    plt.imshow(draw_top_down_map(env.current_map))
    plt.show()
    env.close()


def _expert():
    env = MultiAgentIntersectionEnv(
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
    env = MultiAgentIntersectionEnv(
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
    env = MultiAgentIntersectionEnv(
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
            #
            "use_render": True,
            "debug": True,
            "allow_respawn": False,
            "manual_control": True,
            "num_agents": 20,
            "delay_done": 2,
        }
    )
    o = env.reset()
    total_r = 0
    ep_s = 0
    for i in range(1, 100000):
        actions = {k: [0.0, 1.0] for k in env.vehicles.keys()}
        if len(env.vehicles) == 1:
            actions = {k: [-0, 1.0] for k in env.vehicles.keys()}
        o, r, d, info = env.step(actions)
        for r_ in r.values():
            total_r += r_
        ep_s += 1
        # d.update({"total_r": total_r, "episode length": ep_s})
        # render_text = {
        #     "total_r": total_r,
        #     "episode length": ep_s,
        #     "cam_x": env.main_camera.camera_x,
        #     "cam_y": env.main_camera.camera_y,
        #     "cam_z": env.main_camera.top_down_camera_height,
        #     "alive": len(env.vehicles)
        # }
        # env.render(text=render_text)
        # env.render(mode="top_down")
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
    env = MultiAgentIntersectionEnv({"num_agents": 16})
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
    print(f"(MAIntersection) Total Time Elapse: {time.time() - start}")


def _long_run():
    # Please refer to test_ma_Intersection_reward_done_alignment()
    _out_of_road_penalty = 3
    env = MultiAgentIntersectionEnv(
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


def show_map_and_traj():
    import matplotlib.pyplot as plt
    from metadrive.obs.top_down_renderer import draw_top_down_map, draw_top_down_trajectory
    import json
    import cv2
    import pygame
    env = MultiAgentIntersectionEnv()
    env.reset()
    with open("metasvodist_inter_best.json", "r") as f:
        traj = json.load(f)
    m = draw_top_down_map(env.current_map, simple_draw=False, return_surface=True, reverse_color=True)
    m = draw_top_down_trajectory(
        m, traj, entry_differ_color=True, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    )
    ret = cv2.resize(pygame.surfarray.pixels_red(m), (512, 512), interpolation=cv2.INTER_LINEAR)
    #
    plt.imshow(ret)
    plt.show()
    pygame.image.save(m, "image.png")
    env.close()


if __name__ == "__main__":
    # _draw()
    _vis()
    # _vis_debug_respawn()
    # _profiwdle()
    # _long_run()
    # show_map_and_traj()
    # pygame_replay("parking", MultiAgentParkingLotEnv, False, other_traj="metasvodist_parking_best.json")
    # panda_replay(
    #     "parking",
    #     MultiAgentIntersectionEnv,
    #     False,
    #     other_traj="metasvodist_inter.json",
    #     extra_config={
    #         "global_light": True
    #     }
    # )
    # pygame_replay()
