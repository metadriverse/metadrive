from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils import setup_logger

h_f = 2
w_f = 2

if __name__ == "__main__":
    setup_logger(True)
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            # "traffic_density": 1.0,
            "traffic_mode": "hybrid",
            "start_seed": 82,
            # "debug_physics_world": True,
            # "pstats": True,
            # "show_fps":False,

            # "random_traffic":True,
            "vehicle_config": dict(
                mini_map=(168 * w_f * 6, 84 * h_f * 6, 270),  # buffer length, width
                rgb_camera=(168 * w_f, 84 * h_f),  # buffer length, width
                depth_camera=(168 * w_f, 84 * h_f),  # buffer length, width, view_ground
                show_navi_mark=False,
                wheel_friction=0.6,
                show_lidar=True
            ),
            # "camera_height":100,
            "image_source": "mini_map",
            "manual_control": True,
            "use_render": True,
            "decision_repeat": 5,
            "norm_pixel": True,
            # "debug":True,
            "map_config": {
                BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                BaseMap.GENERATE_CONFIG: "rrXCO",
                BaseMap.LANE_WIDTH: 3.5,
                BaseMap.LANE_NUM: 3,
            }
        }
    )
    o, _ = env.reset()

    depth_camera = env.config["vehicle_config"]["depth_camera"]
    depth_camera = DepthCamera(*depth_camera, chassis_np=env.agent.chassis, engine=env.engine)
    env.agent.add_image_sensor("depth_camera", depth_camera)
    depth_camera.remove_display_region(env.engine)

    # for sensor in env.agent.image_sensors.values():
    #     sensor.remove_display_region(env.engine)
    # env.agent.contact_result_render.detachNode()
    # env.agent.navigation._right_arrow.detachNode()

    env.agent.chassis.setPos(244, 0, 1.5)
    for i in range(1, 100000):
        o, r, tm, tc, info = env.step([0, 1])
        env.render(
            # text={
            #     "vehicle_num": len(env.engine.traffic_manager.traffic_vehicles),
            #     "dist_to_left:": env.agent.dist_to_left,
            #     "dist_to_right:": env.agent.dist_to_right,
            # }
        )
        if tm or tc:
            env.reset()
    env.close()
