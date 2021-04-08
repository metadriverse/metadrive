import math

from pgdrive.envs.pgdrive_env_v2 import PGDriveEnvV2


def test_out_of_road():
    # env = PGDriveEnvV2(dict(vehicle_config=dict(side_detector=dict(num_lasers=8))))
    for steering in [-0.01, 0.01]:
        for distance in [10, 50, 100]:
            env = PGDriveEnvV2(
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
                        points = env.vehicle.side_detector.get_cloud_points()
                        assert min(points) < tolerance, (min(points), tolerance)
                        print(
                            "Side detector minimal distance: {}, Current distance: {}, steering: {}".format(
                                min(points) * distance, distance, steering
                            )
                        )
                        break
            finally:
                env.close()


if __name__ == '__main__':
    test_out_of_road()
