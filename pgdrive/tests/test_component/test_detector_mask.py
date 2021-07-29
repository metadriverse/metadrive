import copy

import numpy as np
from pgdrive.envs import PGDriveEnvV2
from pgdrive.component.vehicle.base_vehicle import BaseVehicle
from pgdrive.component.vehicle_module.distance_detector import DetectorMask
from pgdrive.utils import panda_position


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

            mask_ratio = env.engine.detector_mask.get_mask_ratio()
            print("Mask ratio: ", mask_ratio)
            print("We have: {} vehicles!".format(env.engine.traffic_manager.get_vehicle_num()))

            v = env.vehicle
            v.lidar.perceive(
                v.position,
                v.heading_theta,
                v.engine.physics_world.dynamic_world,
                extra_filter_node={v.chassis_np.node()},
                detector_mask=None
            )
            old_cloud_points = np.array(copy.deepcopy(env.vehicle.lidar.get_cloud_points()))

            position_dict = {}
            heading_dict = {}
            is_target_vehicle_dict = {}
            for v in env.engine.traffic_manager.vehicles:
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
                v.engine.physics_world.dynamic_world,
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


def test_cutils_lidar():
    def _old_perceive(
        self,
        vehicle_position,
        heading_theta,
        physics_world,
        extra_filter_node: set = None,
        detector_mask: np.ndarray = None
    ):
        """
        Call me to update the perception info
        """
        assert detector_mask is not "WRONG"
        # coordinates problem here! take care
        extra_filter_node = extra_filter_node or set()
        pg_start_position = panda_position(vehicle_position, self.height)

        # init
        self.cloud_points.fill(1.0)
        self.detected_objects = []

        # lidar calculation use pg coordinates
        mask = self.mask
        # laser_heading = self._lidar_range + heading_theta
        # point_x = self.perceive_distance * np.cos(laser_heading) + vehicle_position[0]
        # point_y = self.perceive_distance * np.sin(laser_heading) + vehicle_position[1]

        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        for laser_index in range(self.num_lasers):
            # # coordinates problem here! take care

            if (detector_mask is not None) and (not detector_mask[laser_index]):
                # update vis
                if self.cloud_points_vis is not None:
                    laser_end = self._get_laser_end(laser_index, heading_theta, vehicle_position)
                    self._add_cloud_point_vis(laser_index, laser_end)
                continue

            laser_end = self._get_laser_end(laser_index, heading_theta, vehicle_position)
            result = physics_world.rayTestClosest(pg_start_position, laser_end, mask)
            node = result.getNode()
            if node in extra_filter_node:
                # Fall back to all tests.
                results = physics_world.rayTestAll(pg_start_position, laser_end, mask)
                hits = results.getHits()
                hits = sorted(hits, key=lambda ret: ret.getHitFraction())
                for result in hits:
                    if result.getNode() in extra_filter_node:
                        continue
                    self.detected_objects.append(result)
                    self.cloud_points[laser_index] = result.getHitFraction()
                    break
            else:
                hits = result.hasHit()
                self.cloud_points[laser_index] = result.getHitFraction()
                if node:
                    self.detected_objects.append(result)

            # update vis
            if self.cloud_points_vis is not None:
                self._add_cloud_point_vis(laser_index, result.getHitPos() if hits else laser_end)
        return self.cloud_points

    from pgdrive.utils.cutils import _get_fake_cutils
    _fake_cutils = _get_fake_cutils()

    def fake_cutils_perceive(
        self,
        vehicle_position,
        heading_theta,
        physics_world,
        extra_filter_node: set = None,
        detector_mask: np.ndarray = None
    ):
        cloud_points, _, _ = _fake_cutils.cutils_perceive(
            cloud_points=self.cloud_points,
            detector_mask=detector_mask.astype(dtype=np.uint8) if detector_mask is not None else None,
            mask=self.mask,
            lidar_range=self._lidar_range,
            perceive_distance=self.perceive_distance,
            heading_theta=heading_theta,
            vehicle_position_x=vehicle_position[0],
            vehicle_position_y=vehicle_position[1],
            num_lasers=self.num_lasers,
            height=self.height,
            physics_world=physics_world,
            extra_filter_node=extra_filter_node if extra_filter_node else set(),
            require_colors=self.cloud_points_vis is not None,
            ANGLE_FACTOR=self.ANGLE_FACTOR,
            MARK_COLOR0=self.MARK_COLOR[0],
            MARK_COLOR1=self.MARK_COLOR[1],
            MARK_COLOR2=self.MARK_COLOR[2]
        )
        return cloud_points

    env = PGDriveEnvV2({"map": "C", "traffic_density": 1.0, "environment_num": 10})
    try:
        for _ in range(3):
            env.reset()
            ep_count = 0
            for _ in range(3000):
                o, r, d, i = env.step([0, 1])

                v = env.vehicle
                new_cloud_points = v.lidar.perceive(
                    v.position,
                    v.heading_theta,
                    v.engine.physics_world.dynamic_world,
                    extra_filter_node={v.chassis_np.node()},
                    detector_mask=None
                )
                new_cloud_points = np.array(copy.deepcopy(new_cloud_points))
                old_cloud_points = _old_perceive(
                    v.lidar, v.position, v.heading_theta, v.engine.physics_world.dynamic_world, {v.chassis_np.node()},
                    None
                )
                np.testing.assert_almost_equal(new_cloud_points, old_cloud_points)

                fake_cutils_cloud_points = fake_cutils_perceive(
                    v.lidar,
                    v.position,
                    v.heading_theta,
                    v.engine.physics_world.dynamic_world,
                    extra_filter_node={v.chassis_np.node()},
                    detector_mask=None
                )
                np.testing.assert_almost_equal(new_cloud_points, fake_cutils_cloud_points)

                # assert sum(abs(mask.astype(int) - real_mask.astype(int))) <= 3
                v = env.vehicle
                v.lidar.perceive(
                    v.position,
                    v.heading_theta,
                    v.engine.physics_world.dynamic_world,
                    extra_filter_node={v.chassis_np.node()},
                    detector_mask=env.engine.detector_mask.get_mask(v.name)
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
    # test_detector_mask()
    # test_detector_mask_in_lidar()
    test_cutils_lidar()
