from metadrive.component.map.pg_map import PGMap
from metadrive.component.pg_space import Parameter
from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.straight import Straight
from metadrive.component.sensors.lidar import Lidar
from metadrive.constants import PGLineType
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.pg_map_manager import PGMapManager
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

    # We still want to use single-agent observation
    vehicle_config=dict(
        lidar=dict(num_lasers=240, distance=50, num_others=0, gaussian_noise=0.0, dropout_prob=0.0,
                   add_others_navi=False)
    ),
    sensors=dict(lidar=(Lidar, 50)),

    # Number of agents and map setting.
    num_agents=12,
    map_config=dict(lane_num=2, exit_length=60, bottle_lane_num=4, neck_lane_num=1, neck_length=20),

    # Reward setting
    use_lateral=False,

    # Termination condition
    cross_yellow_line_done=False,
    out_of_road_done=False,
    on_continuous_line_done=False,
    out_of_route_done=False,
    crash_done=False,
    max_step_per_agent=3_000,
    horizon=3_000,

    # Debug setting
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
    """Create a complex racing map by manually design the topology."""

    def _generate(self):
        """ Generate the racing map.

        Returns:
            None
        """
        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"

        LANE_NUM = self.config["lane_num"]
        LANE_WIDTH = self.config["lane_width"]

        init_block = FirstPGBlock(
            self.road_network,
            lane_width=LANE_WIDTH,
            lane_num=LANE_NUM,
            render_root_np=parent_node_path,
            physics_world=physics_world,
            remove_negative_lanes=True,
        )
        self.blocks.append(init_block)

        block_s1 = Straight(
            1,
            init_block.get_socket(0),
            self.road_network,
            1,
            remove_negative_lanes=True,
            side_lane_line_type=PGLineType.GUARDRAIL,
            center_line_type=PGLineType.GUARDRAIL
        )
        block_s1.construct_from_config({Parameter.length: 100}, parent_node_path, physics_world)
        self.blocks.append(block_s1)

        block_c1 = Curve(
            2,
            block_s1.get_socket(0),
            self.road_network,
            1,
            remove_negative_lanes=True,
            side_lane_line_type=PGLineType.GUARDRAIL,
            center_line_type=PGLineType.GUARDRAIL
        )
        block_c1.construct_from_config(
            {
                Parameter.length: 200,
                Parameter.radius: 100,
                Parameter.angle: 90,
                Parameter.dir: 1,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c1)

        block_s2 = Straight(
            3,
            block_c1.get_socket(0),
            self.road_network,
            1,
            remove_negative_lanes=True,
            side_lane_line_type=PGLineType.GUARDRAIL,
            center_line_type=PGLineType.GUARDRAIL
        )
        block_s2.construct_from_config({
            Parameter.length: 100,
        }, parent_node_path, physics_world)
        self.blocks.append(block_s2)

        block_c2 = Curve(
            4,
            block_s2.get_socket(0),
            self.road_network,
            1,
            remove_negative_lanes=True,
            side_lane_line_type=PGLineType.GUARDRAIL,
            center_line_type=PGLineType.GUARDRAIL
        )
        block_c2.construct_from_config(
            {
                Parameter.length: 100,
                Parameter.radius: 60,
                Parameter.angle: 90,
                Parameter.dir: 1,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c2)

        block_c3 = Curve(
            5,
            block_c2.get_socket(0),
            self.road_network,
            1,
            remove_negative_lanes=True,
            side_lane_line_type=PGLineType.GUARDRAIL,
            center_line_type=PGLineType.GUARDRAIL
        )
        block_c3.construct_from_config(
            {
                Parameter.length: 100,
                Parameter.radius: 60,
                Parameter.angle: 90,
                Parameter.dir: 1,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c3)

        block_s3 = Straight(
            6,
            block_c3.get_socket(0),
            self.road_network,
            1,
            remove_negative_lanes=True,
            side_lane_line_type=PGLineType.GUARDRAIL,
            center_line_type=PGLineType.GUARDRAIL
        )
        block_s3.construct_from_config({
            Parameter.length: 200,
        }, parent_node_path, physics_world)
        self.blocks.append(block_s3)

        block_c4 = Curve(
            7,
            block_s3.get_socket(0),
            self.road_network,
            1,
            remove_negative_lanes=True,
            side_lane_line_type=PGLineType.GUARDRAIL,
            center_line_type=PGLineType.GUARDRAIL
        )
        block_c4.construct_from_config(
            {
                Parameter.length: 80,
                Parameter.radius: 40,
                Parameter.angle: 90,
                Parameter.dir: 1,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c4)

        block_c5 = Curve(
            8,
            block_c4.get_socket(0),
            self.road_network,
            1,
            remove_negative_lanes=True,
            side_lane_line_type=PGLineType.GUARDRAIL,
            center_line_type=PGLineType.GUARDRAIL
        )
        block_c5.construct_from_config(
            {
                Parameter.length: 40,
                Parameter.radius: 50,
                Parameter.angle: 180,
                Parameter.dir: 1,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c5)

        block_c6 = Curve(
            9,
            block_c5.get_socket(0),
            self.road_network,
            1,
            remove_negative_lanes=True,
            side_lane_line_type=PGLineType.GUARDRAIL,
            center_line_type=PGLineType.GUARDRAIL
        )
        block_c6.construct_from_config(
            {
                Parameter.length: 40,
                Parameter.radius: 50,
                Parameter.angle: 220,
                Parameter.dir: 0,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c6)

        block_c7 = Curve(
            10,
            block_c6.get_socket(0),
            self.road_network,
            1,
            remove_negative_lanes=True,
            side_lane_line_type=PGLineType.GUARDRAIL,
            center_line_type=PGLineType.GUARDRAIL
        )
        block_c7.construct_from_config(
            {
                Parameter.length: 40,
                Parameter.radius: 20,
                Parameter.angle: 180,
                Parameter.dir: 1,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c7)

        block_s4 = Straight(
            11,
            block_c7.get_socket(0),
            self.road_network,
            1,
            remove_negative_lanes=True,
            side_lane_line_type=PGLineType.GUARDRAIL,
            center_line_type=PGLineType.GUARDRAIL
        )
        block_s4.construct_from_config({
            Parameter.length: 100,
        }, parent_node_path, physics_world)
        self.blocks.append(block_s4)

        block_c8 = Curve(
            12,
            block_s4.get_socket(0),
            self.road_network,
            1,
            remove_negative_lanes=True,
            side_lane_line_type=PGLineType.GUARDRAIL,
            center_line_type=PGLineType.GUARDRAIL
        )
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
    """This map manager load the racing map directly, without the burden to manage multiple maps."""

    def __init__(self):
        super(RacingMapManager, self).__init__()

    def reset(self):
        """Only initialize the RacingMap once."""
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(RacingMap, map_config=config["map_config"], random_seed=None)
        else:
            assert len(self.spawned_objects) == 1, "It is supposed to contain one map in this manager"
            _map = self.spawned_objects.values()[0]
        self.load_map(_map)


class MultiAgentRacingEnv(MultiAgentMetaDrive):
    """The Multi-agent Racing Environment"""

    def setup_engine(self):
        """Using the RacingMapManager as the map_manager."""
        super(MultiAgentRacingEnv, self).setup_engine()
        self.engine.update_manager("map_manager", RacingMapManager())

    @staticmethod
    def default_config() -> Config:
        """Use the RACING_CONFIG as the default config."""
        return MultiAgentMetaDrive.default_config().update(RACING_CONFIG, allow_add_new_key=True)


def _vis(generate_video=False):
    """ This function visualize and generate the video for this environment.

    Args:
        generate_video: Whether to generate the videos, in both the top-down view and the third-person view.

    Returns:
        None
    """
    FPS = 60

    env = MultiAgentRacingEnv(dict(
        use_render=generate_video,
        window_size=(1600, 1200),
        num_agents=12,
        # debug=True
    ))
    o, _ = env.reset()

    # env.engine.force_fps.disable()

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

            if generate_video:
                import pygame
                img_interface = env.render(mode="rgb_array")
                img_bev = env.render(
                    mode="topdown",
                    show_agent_name=False,
                    target_vehicle_heading_up=False,
                    draw_target_vehicle_trajectory=False,
                    film_size=(3000, 3000),
                    screen_size=(1000, 1000),
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
                total_r = 0
                print("Reset")
                # env.reset()
                break
    finally:
        env.close()

    if generate_video:
        import datetime
        import os
        import mediapy

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
