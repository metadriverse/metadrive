import math

from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.utils import clip


def test_out_of_road():
    # env = PGDriveEnv(dict(vehicle_config=dict(side_detector=dict(num_lasers=8))))
    for steering in [-0.01, 0.01]:
        for distance in [10, 50, 100]:
            env = PGDriveEnv(
                dict(
                    map="SSSSSSSSSSS",
                    vehicle_config=dict(side_detector=dict(num_lasers=120, distance=distance)),
                    use_render=False,
                    fast=True
                )
            )
            try:
                obs = env.reset()
                tolerance = math.sqrt(env.vehicle.WIDTH**2 + env.vehicle.LENGTH**2) / distance
                for _ in range(100000000):
                    o, r, d, i = env.step([steering, 1])
                    if d:
                        points = \
                            env.vehicle.side_detector.perceive(env.vehicle,
                                                               env.vehicle.engine.physics_world.static_world).cloud_points
                        assert min(points) < tolerance, (min(points), tolerance)
                        print(
                            "Side detector minimal distance: {}, Current distance: {}, steering: {}".format(
                                min(points) * distance, distance, steering
                            )
                        )
                        break
            finally:
                env.close()


def useless_left_right_distance_printing():
    # env = PGDriveEnv(dict(vehicle_config=dict(side_detector=dict(num_lasers=8))))
    for steering in [-0.01, 0.01, -1, 1]:
        # for distance in [10, 50, 100]:
        env = PGDriveEnv(
            dict(
                map="SSSSSSSSSSS",
                vehicle_config=dict(side_detector=dict(num_lasers=0, distance=50)),
                use_render=False,
                fast=True
            )
        )
        try:
            for _ in range(100000000):
                o, r, d, i = env.step([steering, 1])
                vehicle = env.vehicle
                l, r = vehicle.dist_to_left_side, vehicle.dist_to_right_side
                total_width = float(
                    (vehicle.navigation.get_current_lane_num() + 1) * vehicle.navigation.get_current_lane_width()
                )
                print(
                    "Left {}, Right {}, Total {}. Clip Total {}".format(
                        l / total_width, r / total_width, (l + r) / total_width,
                        clip(l / total_width, 0, 1) + clip(r / total_width, 0, 1)
                    )
                )
                if d:
                    break
        finally:
            env.close()


if __name__ == '__main__':
    # test_out_of_road()
    useless_left_right_distance_printing()
