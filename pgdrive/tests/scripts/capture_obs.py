from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.scene_creator.map import Map, MapGenerateMethod
from pgdrive.scene_creator.vehicle_module.depth_camera import DepthCamera
from pgdrive.utils import setup_logger

h_f = 2
w_f = 2


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 1,
                # "traffic_density": 1.0,
                "traffic_mode": "hybrid",
                "start_seed": 82,
                "pg_world_config": {
                    "onscreen_message": True,
                    # "debug_physics_world": True,
                    # "pstats": True,
                    # "show_fps":False,
                },
                # "random_traffic":True,
                "vehicle_config": dict(
                    mini_map=(168 * w_f * 6, 84 * h_f * 6, 270),  # buffer length, width
                    rgb_cam=(168 * w_f, 84 * h_f),  # buffer length, width
                    depth_cam=(168 * w_f, 84 * h_f, True),  # buffer length, width, view_ground
                    show_navi_mark=False,
                    increment_steering=False,
                    wheel_friction=0.6,
                    show_lidar=True
                ),
                # "camera_height":100,
                # "controller":"joystick",
                "image_source": "mini_map",
                "manual_control": True,
                "use_render": True,
                "use_topdown": True,
                "steering_penalty": 0.0,
                "decision_repeat": 5,
                "rgb_clip": True,
                # "debug":True,
                "map_config": {
                    Map.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                    Map.GENERATE_CONFIG: "rrXCO",
                    Map.LANE_WIDTH: 3.5,
                    Map.LANE_NUM: 3,
                }
            }
        )


if __name__ == "__main__":
    setup_logger(True)
    env = TestEnv()
    o = env.reset()

    depth_cam = env.config["vehicle_config"]["depth_cam"]
    depth_cam = DepthCamera(*depth_cam, chassis_np=env.vehicle.chassis_np, pg_world=env.pg_world)
    env.vehicle.add_image_sensor("depth_cam", depth_cam)
    depth_cam.remove_display_region(env.pg_world)

    # for sensor in env.vehicle.image_sensors.values():
    #     sensor.remove_display_region(env.pg_world)
    # env.vehicle.vehicle_panel.remove_display_region(env.pg_world)
    # env.vehicle.collision_info_np.detachNode()
    # env.vehicle.routing_localization._right_arrow.detachNode()

    env.vehicle.chassis_np.setPos(244, 0, 1.5)
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        env.render(
            # text={
            #     "vehicle_num": len(env.scene_manager.traffic_manager.traffic_vehicles),
            #     "dist_to_left:": env.vehicle.dist_to_left,
            #     "dist_to_right:": env.vehicle.dist_to_right,
            #     "env_seed": env.current_map.random_seed
            # }
        )
        if d:
            env.reset()
    env.close()
