from pgdrive.envs.pgdrive_env import PGDriveEnv

from pgdrive.scene_creator.map import Map, MapGenerateMethod
from pgdrive.utils import setup_logger
from pgdrive.scene_creator.ego_vehicle.vehicle_module.depth_camera import DepthCamera

setup_logger(True)

h_f = 2
w_f = 2


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 1,
                "traffic_density": 0.2,
                "traffic_mode": "hybrid",
                "start_seed": 82,
                "pg_world_config": {
                    "onscreen_message": True,
                    # "debug_physics_world": True,
                    "pstats": True
                },
                "vehicle_config": dict(
                    mini_map=(168 * w_f * 6, 84 * h_f * 6, 270),  # buffer length, width
                    rgb_cam=(168 * w_f, 84 * h_f),  # buffer length, width
                    depth_cam=(168 * w_f, 84 * h_f, True),  # buffer length, width, view_ground
                    show_navi_mark=False,
                    increment_steering=False,
                    wheel_friction=0.6,
                    show_lidar=True
                ),
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
                    Map.GENERATE_METHOD: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                    Map.GENERATE_PARA: "rrXCO",
                    Map.LANE_WIDTH: 3.5,
                    Map.LANE_NUM: 3,
                }
            }
        )


if __name__ == "__main__":
    env = TestEnv()
    o = env.reset()

    depth_cam = env.config["vehicle_config"]["depth_cam"]
    depth_cam = DepthCamera(*depth_cam, chassis_np=env.vehicle.chassis_np, pg_world=env.pg_world)
    env.vehicle.add_image_sensor("depth_cam", depth_cam)
    depth_cam.remove_display_region(env.pg_world)

    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        env.render(
            # text={
            #     "vehicle_num": len(env.scene_manager.traffic_mgr.traffic_vehicles),
            #     "dist_to_left:": env.vehicle.dist_to_left,
            #     "dist_to_right:": env.vehicle.dist_to_right,
            #     "env_seed": env.current_map.random_seed
            # }
        )
        if d:
            env.reset()
    env.close()
