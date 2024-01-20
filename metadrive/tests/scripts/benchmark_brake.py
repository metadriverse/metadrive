import time

import numpy as np

from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.envs import MetaDriveEnv


def get_result(env):
    obs, _ = env.reset()
    start = time.time()
    max_speed_km_h = 0.0
    reported_max_speed = None
    reported_end = None
    reported_start = None
    reported_rotation = None
    start_heading = env.agent.heading_theta
    rotate_start_pos = None
    max_speed_loc = None
    for s in range(10000):
        if s < 20:
            action = np.array([0.0, 0.0])
        elif env.agent.speed_km_h < 100 and not reported_max_speed:
            action = np.array([0.0, 1.0])
        else:
            action = np.array([0.0, -1.0])
            # action = np.array([0.0, 0.0])

        if s > 20 and env.agent.speed_km_h > 1.0 and not reported_start:
            # print("Start the car at {}".format(s))
            reported_start = s
            start_time = time.time()

        if s > 20 and env.agent.speed_km_h >= 100 and not reported_max_speed:
            spend = (s - 1 - reported_start) * 0.1
            print(
                "Achieve max speed: {} at {}. Spend {} s. Current location: {}".format(
                    max_speed_km_h, s - 1, spend, env.agent.position
                )
            )
            # print("real time spend to acc: {}".format(time.time() - start_time))
            reported_max_speed = s
            max_speed_loc = env.agent.position

        max_speed_km_h = max(max_speed_km_h, env.agent.speed_km_h)

        if s > 20 and env.agent.speed_km_h <= 1.0 and reported_max_speed and not reported_end:
            dist = env.agent.position - max_speed_loc
            dist = dist[0]
            # print("Stop the car at {}. Distance {}. Current location: {}".format(s, dist, env.agent.position))
            reported_end = True

        speed = env.agent.speed_km_h
        current_heading = env.agent.heading_theta
        if reported_end and not reported_rotation:
            if rotate_start_pos is None:
                rotate_start_pos = env.agent.position
            if speed < 99:
                action = np.array([0.0, 1.0])
            else:
                action = np.array([-1.0, 1.0])
                if abs(current_heading - start_heading) >= np.pi / 2:
                    rotate_displacement = np.asarray(env.agent.position) - np.asarray(rotate_start_pos)
                    reported_rotation = True

        o, r, tm, tc, i = env.step(action)

        if reported_max_speed and reported_start and reported_end and reported_rotation:
            break
        # env.render()
    return spend, dist, rotate_displacement


if __name__ == '__main__':

    for friction in [0.8]:
        env = MetaDriveEnv(
            {
                "num_scenarios": 1,
                "traffic_density": 0.0,
                "start_seed": 4,
                "manual_control": True,
                "use_render": True,
                "image_observation": True,
                "vehicle_config": {
                    "max_engine_force": 1000,
                    "max_brake_force": 100,
                    "max_steering": 40,
                    "wheel_friction": friction
                },
                "map_config": {
                    BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                    BaseMap.GENERATE_CONFIG: "SSSSSSSSSS"
                }
            }
        )
        acc_time, brake_dist, rotate_dis = get_result(env)
        print(
            "Friction {}. Acceleration time: {:.3f}. Brake distance: {:.3f}. Rotation 90 degree displacement X: {:.3f}, Y: {:.3f}\n\n"
            .format(friction, acc_time, brake_dist, rotate_dis[0], rotate_dis[1])
        )
        env.close()
