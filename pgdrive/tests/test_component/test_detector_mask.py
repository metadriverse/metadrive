import copy

import numpy as np
from pgdrive.envs import PGDriveEnvV2
from pgdrive.scene_creator.vehicle.base_vehicle import BaseVehicle
from pgdrive.scene_creator.vehicle_module.distance_detector import DetectorMask


def _line_intersect(theta, center, point1, point2, maximum: float = 10000):
    """
    theta: the direction of the laser
    center: the center of the source of laser
    point1: one point of the line intersection
    point2: another poing of the line intersection
    maximum: the return value if no intersection
    """
    x0, y0 = center
    x1, y1 = point1
    x2, y2 = point2
    dx1 = np.cos(theta)
    dy1 = np.sin(theta)
    dx2 = x2 - x1
    dy2 = y2 - y1
    DET = -dx1 * dy2 + dy1 * dx2
    if abs(DET) < 1e-9:
        return maximum
    r = 1.0 / DET * (-dy2 * (x1 - x0) + dx2 * (y1 - y0))
    s = 1.0 / DET * (-dy1 * (x1 - x0) + dx1 * (y1 - y0))
    if 0 - 1e-5 < s < 1 + 1e-5 and r >= 0:
        return r
    return maximum


def _search_angle(point1, point2, num_lasers, start, heading, perceive_distance: float = 50):
    laser_heading = np.arange(0, num_lasers) * 2 * np.pi / num_lasers + heading
    result = []
    for laser_index in range(num_lasers):
        ret = _line_intersect(theta=laser_heading[laser_index], center=start, point1=point1, point2=point2, maximum=1e8)
        assert 0 <= ret < 1e9
        if ret <= 1e8:
            result.append(ret / 1e8)
        else:
            result.append(1.0)
    return np.asarray(result)


def _test_mask(mask, stick_1_heading_deg, stick_2_heading_deg, max_span, stick1_x, stick2_x):
    stick_1_heading = np.deg2rad(stick_1_heading_deg)
    stick_2_heading = np.deg2rad(stick_2_heading_deg)
    stick_1_heading = stick_1_heading % (2 * np.pi)
    stick_2_heading = stick_2_heading % (2 * np.pi)

    stick1_pos = (stick1_x, 0)
    stick2_pos = (stick2_x, 0)
    mask.update_mask(
        position_dict={
            "stick1": stick1_pos,
            "stick2": stick2_pos
        },
        heading_dict={
            "stick1": stick_1_heading,
            "stick2": stick_2_heading
        },
        is_target_vehicle_dict={
            "stick1": True,
            "stick2": True
        }
    )
    mask_1 = mask.get_mask("stick1")
    mask_2 = mask.get_mask("stick2")

    left_of_stick2 = (stick2_pos[0], -max_span / 2)
    right_of_stick2 = (stick2_pos[0], max_span / 2)

    real_mask_1 = _search_angle(
        point1=left_of_stick2,
        point2=right_of_stick2,
        num_lasers=360,
        start=stick1_pos,
        heading=stick_1_heading,
        perceive_distance=100 * max_span
    ) < 1.0
    res = np.stack([real_mask_1, mask_1])
    assert all(mask_1[real_mask_1])  # mask 1 should at least include all True of real mask.
    if abs(stick1_x - stick2_x) > max_span:
        assert sum(abs(mask_1.astype(int) - real_mask_1.astype(int))) <= 3

    left_of_stick1 = (stick1_pos[0], -max_span / 2)
    right_of_stick1 = (stick1_pos[0], max_span / 2)

    real_mask_2 = _search_angle(
        point1=left_of_stick1,
        point2=right_of_stick1,
        num_lasers=360,
        start=stick2_pos,
        heading=stick_2_heading,
        perceive_distance=100 * max_span
    ) < 1.0
    res2 = np.stack([real_mask_2, mask_2])
    assert all(mask_2[real_mask_2])  # mask 1 should at least include all True of real mask.
    if abs(stick1_x - stick2_x) > max_span:
        assert sum(abs(mask_2.astype(int) - real_mask_2.astype(int))) <= 3


def test_detector_mask():
    # A infinite long (1e7) stick 2 in front of (0.01m) stick 1.

    pos_xy = [0, -1, 1, -100, 100]
    angles = [0, 0.01, 30, 89, 90, 91, 130, 180, 181, 270, 360, 400]
    angles += [-a for a in angles]

    mask = DetectorMask(num_lasers=360, max_span=1e7)
    _test_mask(mask, -270, 30, 1e7, 0, -1)

    mask = DetectorMask(num_lasers=360, max_span=1)
    _test_mask(mask, -180, -300, 1, 0, -1)
    _test_mask(mask, -361, 270, 1, 0, -100)

    for max_span in [1e7, 1, 0.1]:
        mask = DetectorMask(num_lasers=360, max_span=max_span)
        for pos1_x in pos_xy:
            for pos2_x in pos_xy:
                angles1 = np.random.choice(angles, 5)
                angles2 = np.random.choice(angles, 5)
                for stick_1_heading_deg in angles1:
                    for stick_2_heading_deg in angles2:
                        _test_mask(mask, stick_1_heading_deg, stick_2_heading_deg, max_span, pos1_x, pos2_x)
                print("Finish. ", max_span, pos1_x, pos2_x)


def test_detector_mask_in_lidar():
    env = PGDriveEnvV2({"traffic_density": 1.0, "map": "SSSSS", "random_traffic": False})
    try:
        env.reset()
        span = 2 * max(env.vehicle.WIDTH, env.vehicle.LENGTH)
        detector_mask = DetectorMask(
            env.config.vehicle_config.lidar.num_lasers, span, max_distance=env.config.vehicle_config.lidar.distance
        )
        ep_count = 0
        for _ in range(3000):
            o, r, d, i = env.step([0, 1])

            mask_ratio = env.scene_manager.detector_mask.get_mask_ratio()
            print("Mask ratio: ", mask_ratio)
            print("We have: {} vehicles!".format(env.scene_manager.traffic_mgr.get_vehicle_num()))

            v = env.vehicle
            v.lidar.perceive(
                v.position,
                v.heading_theta,
                v.pg_world.physics_world.dynamic_world,
                extra_filter_node={v.chassis_np.node()},
                detector_mask=None
            )
            old_cloud_points = np.array(copy.deepcopy(env.vehicle.lidar.get_cloud_points()))

            position_dict = {}
            heading_dict = {}
            is_target_vehicle_dict = {}
            for v in env.scene_manager.traffic_mgr.vehicles:
                position_dict[v.name] = v.position
                heading_dict[v.name] = v.heading_theta
                is_target_vehicle_dict[v.name] = True if isinstance(v, BaseVehicle) else False

            detector_mask.update_mask(
                position_dict=position_dict, heading_dict=heading_dict, is_target_vehicle_dict=is_target_vehicle_dict
            )

            real_mask = old_cloud_points != 1.0
            mask = detector_mask.get_mask(env.vehicle.name)
            stack = np.stack([old_cloud_points, real_mask, mask])
            if not all(mask[real_mask]):
                print('stop')
            assert all(mask[real_mask])  # mask 1 should at least include all True of real mask.

            print(
                "Num of true in our mask: {}, in old mask: {}. Overlap: {}. We have {} more.".format(
                    sum(mask.astype(int)), sum(real_mask.astype(int)), sum(mask[real_mask].astype(int)),
                    sum(mask.astype(int)) - sum(real_mask.astype(int))
                )
            )

            # assert sum(abs(mask.astype(int) - real_mask.astype(int))) <= 3
            v = env.vehicle
            v.lidar.perceive(
                v.position,
                v.heading_theta,
                v.pg_world.physics_world.dynamic_world,
                extra_filter_node={v.chassis_np.node()},
                detector_mask=mask
            )
            new_cloud_points = np.array(copy.deepcopy(env.vehicle.lidar.get_cloud_points()))
            np.testing.assert_almost_equal(old_cloud_points, new_cloud_points)

            if d:
                env.reset()
                ep_count += 1
                if ep_count == 3:
                    break
    finally:
        env.close()


if __name__ == '__main__':
    test_detector_mask()
    test_detector_mask_in_lidar()
