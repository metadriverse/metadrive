import time

import numpy as np

from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.scene_creator.map import Map, MapGenerateMethod


def get_result(env):
    obs = env.reset()
    start = time.time()
    max_speed = 0.0
    reported_max_speed = None
    reported_end = None
    reported_start = None
    reported_rotation = None
    start_heading = env.vehicle.heading_theta
    rotate_start_pos = None
    max_speed_loc = None
    for s in range(10000):
        if s < 20:
            action = np.array([0.0, 0.0])
        elif env.vehicle.speed < 100 and not reported_max_speed:
            action = np.array([0.0, 1.0])
        else:
            action = np.array([0.0, -1.0])
            # action = np.array([0.0, 0.0])

        if s > 20 and env.vehicle.speed > 1.0 and not reported_start:
            print("Start the car at {}".format(s))
            reported_start = s
            start_time = time.time()

        if s > 20 and env.vehicle.speed >= 100 and not reported_max_speed:
            spend = (s - 1 - reported_start) * 0.1
            print(
                "Achieve max speed: {} at {}. Spend {} s. Current location: {}".format(
                    max_speed, s - 1, spend, env.vehicle.position
                )
            )
            print("real time spend to acc: {}".format(time.time() - start_time))
            reported_max_speed = s
            max_speed_loc = env.vehicle.position

        max_speed = max(max_speed, env.vehicle.speed)

        if s > 20 and env.vehicle.speed <= 0.1 and reported_max_speed and not reported_end:
            dist = env.vehicle.position - max_speed_loc
            dist = dist[0]
            print("Stop the car at {}. Distance {}. Current location: {}".format(s, dist, env.vehicle.position))
            reported_end = True

        speed = env.vehicle.speed
        current_heading = env.vehicle.heading_theta
        if reported_end and not reported_rotation:
            if rotate_start_pos is None:
                rotate_start_pos = env.vehicle.position
            if speed < 99:
                action = np.array([0.0, 1.0])
            else:
                action = np.array([-1.0, 1.0])
                if abs(current_heading - start_heading) >= np.pi / 2:
                    rotate_displacement = np.asarray(env.vehicle.position) - np.asarray(rotate_start_pos)
                    reported_rotation = True

        o, r, d, i = env.step(action)

        if reported_max_speed and reported_start and reported_end and reported_rotation:
            break
        env.render()
    return spend, dist, rotate_displacement


if __name__ == '__main__':

    for friction in [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0]:
        # for friction in [0.9, 1.0, 1.1]:
        env = PGDriveEnv(
            {
                "environment_num": 1,
                "traffic_density": 0.0,
                "start_seed": 4,
                "pg_world_config": {
                    "debug": False,
                },
                "manual_control": False,
                "use_render": True,
                "use_image": True,
                "map_config": {
                    Map.GENERATE_METHOD: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                    Map.GENERATE_PARA: "SSSSSSSSSS"
                }
            }
        )
        acc_time, brake_dist, rotate_dis = get_result(env)
        print(
            "Friction {}. Acceleration time: {:.3f}. Brake distance: {:.3f}. Rotation 90 degree displacement X: {:.3f}, Y: {:.3f}"
            .format(friction, acc_time, brake_dist, rotate_dis[0], rotate_dis[1])
        )
        env.close()
