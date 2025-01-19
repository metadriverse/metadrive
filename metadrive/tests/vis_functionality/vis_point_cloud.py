from metadrive.component.sensors.point_cloud_lidar import PointCloudLidar
import numpy as np
from panda3d.core import Point3
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv

if __name__ == "__main__":
    env = SafeMetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "accident_prob": 1.,
            "start_seed": 4,
            "map": "CSCSCCCCCC",
            "manual_control": True,
            # "use_render": True,
            "image_observation": True,
            "map_region_size": 2048,
            # "norm_pixel": True,
            "use_render": True,
            "use_mesh_terrain": True,
            "debug": True,
            "interface_panel": ["point_cloud"],
            "sensors": dict(point_cloud=(PointCloudLidar, 200, 64, True)),  # 64 channel lidar
            "vehicle_config": dict(image_source="point_cloud"),
            # "map_config": {
            #     BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
            #     BaseMap.GENERATE_CONFIG: 12,
            #     BaseMap.LANE_WIDTH: 3.5,
            #     BaseMap.LANE_NUM: 3,
            # }
        }
    )
    env.reset()
    drawer = env.engine.make_line_drawer()
    cam = env.engine.get_sensor("point_cloud").cam
    env.engine.accept("9", env.engine.terrain.reload_terrain_shader)

    for i in range(1, 100000):
        o, r, tm, tc, info = env.step([0, 1])
        assert env.observation_space.contains(o)

        # to world coordinate
        points = o["image"][..., :, -1] + np.asarray(env.engine.render.get_relative_point(cam, Point3(0, 0, 0)))

        drawer.reset()
        drawer.draw_lines(points)
        # drawer.draw_points(points, colors=[(0, 0, 1, 1)] * len(points))
        #
        # np.zeros([60, 3])
        #
        # env.vehicle.convert_to_world_coordinates()
        if env.config["use_render"]:
            # for i in range(ImageObservation.STACK_SIZE):
            #     ObservationType.show_gray_scale_array(o["image"][:, :, i])
            env.render()
        # if tm or tc:
        #     # print("Reset")
        #     env.reset()
    env.close()
