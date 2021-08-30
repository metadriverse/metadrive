import logging
import math


def _get_fake_cutils():
    class FakeCutils:
        @classmethod
        def cutils_norm(cls, x, y):
            return math.sqrt(x**2 + y**2)

        @classmethod
        def cutils_clip(cls, a, low, high):
            return min(max(a, low), high)

        @classmethod
        def cutils_panda_position(cls, position_x, position_y, z=0.0):
            return position_x, -position_y, z

        @classmethod
        def cutils_add_cloud_point_vis(
            cls, point_x, point_y, height, num_lasers, laser_index, ANGLE_FACTOR, MARK_COLOR0, MARK_COLOR1, MARK_COLOR2
        ):
            f = laser_index / num_lasers if ANGLE_FACTOR else 1
            return laser_index, (point_x, point_y, height), (f * MARK_COLOR0, f * MARK_COLOR1, f * MARK_COLOR2)

        @classmethod
        def cutils_get_laser_end(
            cls, lidar_range, perceive_distance, laser_index, heading_theta, vehicle_position_x, vehicle_position_y
        ):
            return (
                perceive_distance * math.cos(lidar_range[laser_index] + heading_theta) + vehicle_position_x,
                perceive_distance * math.sin(lidar_range[laser_index] + heading_theta) + vehicle_position_y
            )

        @classmethod
        def cutils_perceive(
            cls, cloud_points, detector_mask, mask, lidar_range, perceive_distance, heading_theta, vehicle_position_x,
            vehicle_position_y, num_lasers, height, physics_world, extra_filter_node, require_colors, ANGLE_FACTOR,
            MARK_COLOR0, MARK_COLOR1, MARK_COLOR2
        ):
            """A naive re-implement of code in cutils.pyx"""
            cloud_points.fill(1.0)
            detected_objects = []
            colors = []
            pg_start_position = cls.cutils_panda_position(vehicle_position_x, vehicle_position_y, height)

            for laser_index in range(num_lasers):
                if (detector_mask is not None) and (not detector_mask[laser_index]):
                    # update vis
                    if require_colors:
                        point_x, point_y = cls.cutils_get_laser_end(
                            lidar_range, perceive_distance, laser_index, heading_theta, vehicle_position_x,
                            vehicle_position_y
                        )
                        point_x, point_y, point_z = cls.cutils_panda_position(point_x, point_y, height)
                        colors.append(
                            cls.cutils_add_cloud_point_vis(
                                point_x, point_y, height, num_lasers, laser_index, ANGLE_FACTOR, MARK_COLOR0,
                                MARK_COLOR1, MARK_COLOR2
                            )
                        )
                    continue

                # # coordinates problem here! take care
                point_x, point_y = cls.cutils_get_laser_end(
                    lidar_range, perceive_distance, laser_index, heading_theta, vehicle_position_x, vehicle_position_y
                )
                laser_end = cls.cutils_panda_position(point_x, point_y, height)
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
                        detected_objects.append(result)
                        cloud_points[laser_index] = result.getHitFraction()
                        laser_end = result.getHitPos()
                        break
                else:
                    cloud_points[laser_index] = result.getHitFraction()
                    if result.hasHit():
                        laser_end = result.getHitPos()
                    if node:
                        detected_objects.append(result)
                if require_colors:
                    colors.append(
                        cls.cutils_add_cloud_point_vis(
                            laser_end[0], laser_end[1], height, num_lasers, laser_index, ANGLE_FACTOR, MARK_COLOR0,
                            MARK_COLOR1, MARK_COLOR2
                        )
                    )

            return cloud_points, detected_objects, colors

    return FakeCutils


def import_cutils(use_fake_cutils=False):
    if use_fake_cutils:
        return _get_fake_cutils()
    try:
        from metadrive import cutils
    except:
        msg = (
            "It seems you don't install our cython utilities yet! Please reinstall MetaDrive via: "
            "<pip install -e .> or <pip install metadrive>! But don't worry! We are now falling back to native "
            "python implementation!"
        )
        print(msg)
        logging.warning(msg)
        cutils = _get_fake_cutils()
    return cutils
