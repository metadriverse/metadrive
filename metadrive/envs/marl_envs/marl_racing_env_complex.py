from collections import defaultdict

from metadrive.component.map.pg_map import PGMap
from metadrive.component.pg_space import Parameter
from metadrive.component.pgblock.curve import CurveWithGuardrail
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.straight import StraightWithGuardrail
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.utils import Config

RACING_CONFIG = dict(

    # Misc.
    use_render=False,
    manual_control=False,
    traffic_density=0,
    random_agent_model=False,
    top_down_camera_initial_x=95,
    top_down_camera_initial_y=15,
    top_down_camera_initial_z=120,
    allow_respawn=False,
    vehicle_config=dict(show_lidar=False, show_navi_mark=False),

    # Number of agents and map setting.
    num_agents=9,
    map_config=dict(lane_num=3, exit_length=40, bottle_lane_num=4, neck_lane_num=1, neck_length=20),

    # Reward setting
    use_lateral=False,

    # Termination condition
    cross_yellow_line_done=False,
    out_of_road_done=False,
    on_continuous_line_done=False,
    out_of_route_done=False,
    crash_done=False,
    max_step_per_agent=5_000,

    # Debug setting
    agent_policy=IDMPolicy,
    show_fps=False,
    use_chase_camera_follow_lane=True,
    camera_smooth_buffer_size=100,

    show_interface=False,

    camera_dist=10,
    camera_pitch=15,
    camera_height=6,

    # debug=True,
)


class RacingMap(PGMap):
    def _generate(self):
        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"

        LANE_NUM = self.config["lane_num"]
        LANE_WIDTH = self.config["lane_width"]
        # self.config.update({"bottle_lane_num", 4})

        # test = TestBlock(False)
        # initialize_asset_loader(engine=test)
        # global_network = NodeRoadNetwork()
        blocks = []

        init_block = FirstPGBlock(
            self.road_network,
            lane_width=LANE_WIDTH,
            lane_num=LANE_NUM,
            render_root_np=parent_node_path,
            physics_world=physics_world,
            # length=1,
            ignore_adverse_road=True,
        )
        self.blocks.append(init_block)

        # block_s1 = StraightWithGuardrail(1, init_block.get_socket(0), self.road_network, 1)
        # block_s1.construct_from_config(
        #     {
        #         Parameter.length: 100
        #     }, parent_node_path, physics_world
        # )
        # self.blocks.append(block_s1)

        block_s1 = StraightWithGuardrail(1, init_block.get_socket(0), self.road_network, 1)
        block_s1.construct_from_config({Parameter.length: 100}, parent_node_path, physics_world)
        self.blocks.append(block_s1)

        block_c1 = CurveWithGuardrail(2, block_s1.get_socket(0), self.road_network, 1)
        block_c1.construct_from_config(
            {
                Parameter.length: 200,
                Parameter.radius: 100,
                Parameter.angle: 90,
                Parameter.dir: 1,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c1)

        block_s2 = StraightWithGuardrail(3, block_c1.get_socket(0), self.road_network, 1)
        block_s2.construct_from_config({
            Parameter.length: 100,
        }, parent_node_path, physics_world)
        self.blocks.append(block_s2)

        block_c2 = CurveWithGuardrail(4, block_s2.get_socket(0), self.road_network, 1)
        block_c2.construct_from_config(
            {
                Parameter.length: 100,
                Parameter.radius: 60,
                Parameter.angle: 90,
                Parameter.dir: 1,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c2)

        block_c3 = CurveWithGuardrail(5, block_c2.get_socket(0), self.road_network, 1)
        block_c3.construct_from_config(
            {
                Parameter.length: 100,
                Parameter.radius: 60,
                Parameter.angle: 90,
                Parameter.dir: 1,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c3)

        block_s3 = StraightWithGuardrail(6, block_c3.get_socket(0), self.road_network, 1)
        block_s3.construct_from_config({
            Parameter.length: 200,
        }, parent_node_path, physics_world)
        self.blocks.append(block_s3)

        # last_block = block_c3

        # Build Bottleneck
        # merge = Merge(
        #     1, last_block.get_socket(index=0), self.road_network, random_seed=1, ignore_intersection_checking=False,
        #     remove_negative_lanes=True
        # )
        # merge.construct_from_config(
        #     dict(
        #         lane_num=self.config["bottle_lane_num"] - self.config["neck_lane_num"],
        #         length=self.config["neck_length"]
        #     ), parent_node_path, physics_world
        # )
        # self.blocks.append(merge)
        # split = Split(
        #     2, merge.get_socket(index=0), self.road_network, random_seed=1, ignore_intersection_checking=False,
        #     remove_negative_lanes=True
        # )
        # split.construct_from_config(
        #     {
        #         "length": 100,
        #         "lane_num": LANE_NUM
        #     }, parent_node_path, physics_world
        # )
        # self.blocks.append(split)

        # block_s3 = split

        block_c4 = CurveWithGuardrail(7, block_s3.get_socket(0), self.road_network, 1)
        block_c4.construct_from_config(
            {
                Parameter.length: 80,
                Parameter.radius: 40,
                Parameter.angle: 90,
                Parameter.dir: 1,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c4)

        block_c5 = CurveWithGuardrail(8, block_c4.get_socket(0), self.road_network, 1)
        block_c5.construct_from_config(
            {
                Parameter.length: 40,
                Parameter.radius: 50,
                Parameter.angle: 180,
                Parameter.dir: 1,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c5)

        block_c6 = CurveWithGuardrail(9, block_c5.get_socket(0), self.road_network, 1)
        block_c6.construct_from_config(
            {
                Parameter.length: 40,
                Parameter.radius: 50,
                Parameter.angle: 220,
                Parameter.dir: 0,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c6)

        block_c7 = CurveWithGuardrail(10, block_c6.get_socket(0), self.road_network, 1)
        block_c7.construct_from_config(
            {
                Parameter.length: 40,
                Parameter.radius: 20,
                Parameter.angle: 180,
                Parameter.dir: 1,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c7)

        block_s4 = StraightWithGuardrail(11, block_c7.get_socket(0), self.road_network, 1)
        block_s4.construct_from_config({
            Parameter.length: 100,
        }, parent_node_path, physics_world)
        self.blocks.append(block_s4)

        block_c8 = CurveWithGuardrail(12, block_s4.get_socket(0), self.road_network, 1)
        block_c8.construct_from_config(
            {
                Parameter.length: 100,
                Parameter.radius: 40,
                Parameter.angle: 140,
                Parameter.dir: 0,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c8)


class RacingMapManager(PGMapManager):
    def __init__(self):
        super(RacingMapManager, self).__init__()

    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(RacingMap, map_config=config["map_config"], random_seed=None)
        else:
            assert len(self.spawned_objects) == 1, "It is supposed to contain one map in this manager"
            _map = self.spawned_objects.values()[0]
        self.load_map(_map)


class MultiAgentRacingEnv(MultiAgentMetaDrive):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._completed_loop_count = defaultdict(int)

    def setup_engine(self):
        super(MultiAgentRacingEnv, self).setup_engine()
        self.engine.update_manager("map_manager", RacingMapManager())

    @staticmethod
    def default_config() -> Config:
        return MultiAgentMetaDrive.default_config().update(RACING_CONFIG, allow_add_new_key=True)

    @property
    def real_destination(self):
        return list(self.current_map.road_network.graph.keys())[-1]

    @property
    def fake_destination(self):
        return list(self.current_map.road_network.graph.keys())[-2]

    # def _is_arrive_destination(self, vehicle):
    #     """
    #     Args:
    #         vehicle: The BaseVehicle instance.
    #
    #     Returns:
    #         flag: Whether this vehicle arrives its destination.
    #     """
    #     flag = super()._is_arrive_destination(vehicle)
    #     if flag:
    #         if vehicle.config["destination"] == self.fake_destination:
    #             vehicle.config["destination"] = self.real_destination
    #         else:
    #             vehicle.config["destination"] = self.fake_destination
    #         vehicle.reset_navigation()
    #         flag = False
    #     return flag

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
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        reward = 0.0
        reward += self.config["driving_reward"] * (long_now - long_last)
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


def _vis(generate_video=False):
    FPS = 60

    env = MultiAgentRacingEnv(dict(
        use_render=generate_video,
        window_size=(1600, 1200),
        max_step_per_agent=3_000,
        horizon=3_000,

        debug=True
    ))
    o, _ = env.reset()
    env.engine.force_fps.disable()

    for v in env.vehicles.values():
        v.expert_takeover = True

    total_r = 0
    ep_s = 0
    video_bev = []
    video_interface = []

    try:
        for i in range(1, 100000):
            o, r, tm, tc, info = env.step({k: [-0.05, 1.0] for k in env.vehicles.keys()})
            for r_ in r.values():
                total_r += r_
            ep_s += 1

            env.render(mode="topdown")

            if generate_video:
                import mediapy
                import pygame
                img_interface = env.render(mode="rgb_array")
                img_bev = env.render(
                    mode="topdown",
                    show_agent_name=False,
                    target_vehicle_heading_up=False,
                    draw_target_vehicle_trajectory=False,
                    film_size=(2000, 2000),
                    screen_size=(2000, 2000),
                    crash_vehicle_done=False,
                )
                img_bev = pygame.surfarray.array3d(img_bev)
                img_bev = img_bev.swapaxes(0, 1)
                img_bev = img_bev[::-1]
                video_bev.append(img_bev)
                video_interface.append(img_interface)

            if tm["__all__"]:
                print(
                    f"Finish! Current step {i}. Group Reward: {total_r}. Average reward: {total_r / env.agent_manager.next_agent_count}"
                )
                # total_r = 0
                print("Reset")
                # env.reset()
                break
    finally:
        env.close()

    if generate_video:
        import datetime
        import os

        folder_name = "marl_racing_video_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(folder_name, exist_ok=True)

        video_base_name = f"{folder_name}/video"
        video_name_bev = video_base_name + "_bev.mp4"
        print("BEV video should be saved at: ", video_name_bev)
        mediapy.write_video(video_name_bev, video_bev, fps=FPS)

        video_name_interface = video_base_name + "_interface.mp4"
        print("Interface video should be saved at: ", video_name_interface)
        mediapy.write_video(video_name_interface, video_interface, fps=FPS)


if __name__ == "__main__":
    _vis(generate_video=True)
