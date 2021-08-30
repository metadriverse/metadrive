# Build via: python setup.py build_ext --inplace
cimport numpy as cnp
import cython

ctypedef cnp.float64_t np_float64_t
ctypedef cnp.npy_bool np_bool_t
ctypedef cnp.int64_t np_int64_t
ctypedef cnp.uint8_t np_uint8_t
from cpython cimport bool as bool_t, set as set_t, tuple as tuple_t, list as list_t

cdef extern from "math.h":
    double sqrt(double x)
    double sin(double x)
    double cos(double x)
    double acos(double x)
    double fabs(double x)
    double atan2(double y, double x)
    double asin(double x)
    double sqrt(double x)
    double tan(double x)
    int floor(double x)
    int ceil(double x)
    double fmin(double x, double y)
    double fmax(double x, double y)

def cutils_panda_position(np_float64_t position_x, np_float64_t position_y, np_float64_t z=0.0):
    return position_x, -position_y, z

def cutils_add_cloud_point_vis(
        np_float64_t point_x,
        np_float64_t point_y,
        np_float64_t height,
        np_float64_t num_lasers,
        int laser_index,
        bool_t ANGLE_FACTOR,
        np_float64_t MARK_COLOR0,
        np_float64_t MARK_COLOR1,
        np_float64_t MARK_COLOR2
):
    cdef np_float64_t f = laser_index / num_lasers if ANGLE_FACTOR else 1
    return laser_index, (point_x, point_y, height), (f * MARK_COLOR0, f * MARK_COLOR1, f * MARK_COLOR2)

def cutils_get_laser_end(
        cnp.ndarray[np_float64_t, ndim=1] lidar_range,
        np_float64_t perceive_distance,
        int laser_index,
        np_float64_t heading_theta,
        np_float64_t vehicle_position_x,
        np_float64_t vehicle_position_y
):
    return (
        perceive_distance * cos(lidar_range[laser_index] + heading_theta) + vehicle_position_x,
        perceive_distance * sin(lidar_range[laser_index] + heading_theta) + vehicle_position_y
    )

# Remove this check to further accelerate. But this might cause fatal error! So I just commented them out here.
# @cython.wraparound(False)
# @cython.cdivision(True)
# @cython.nonecheck(False)
def cutils_perceive(
        cnp.ndarray[np_float64_t, ndim=1] cloud_points,
        cnp.ndarray[np_uint8_t, ndim=1] detector_mask,
        mask,
        cnp.ndarray[np_float64_t, ndim=1] lidar_range,
        np_float64_t perceive_distance,
        np_float64_t heading_theta,
        np_float64_t vehicle_position_x,
        np_float64_t vehicle_position_y,
        int num_lasers,
        np_float64_t height,
        physics_world,
        set_t extra_filter_node,
        bool_t require_colors,
        bool_t ANGLE_FACTOR,
        np_float64_t MARK_COLOR0,
        np_float64_t MARK_COLOR1,
        np_float64_t MARK_COLOR2
):
    # init
    cloud_points.fill(1.0)
    cdef list_t detected_objects = []
    cdef list_t colors = []
    cdef tuple_t pg_start_position = cutils_panda_position(vehicle_position_x, vehicle_position_y, height)
    cdef np_float64_t point_x = 0.0, point_y = 0.0, point_z = 0.0  # useless

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    for laser_index in range(num_lasers):
        if (detector_mask is not None) and (not detector_mask[laser_index]):
            # update vis
            if require_colors:
                point_x, point_y = cutils_get_laser_end(
                    lidar_range, perceive_distance, laser_index, heading_theta, vehicle_position_x,
                    vehicle_position_y
                )
                point_x, point_y, point_z = cutils_panda_position(point_x, point_y, height)
                colors.append(cutils_add_cloud_point_vis(
                    point_x, point_y, height, num_lasers, laser_index, ANGLE_FACTOR, MARK_COLOR0, MARK_COLOR1,
                    MARK_COLOR2)
                )
            continue

        # # coordinates problem here! take care
        point_x, point_y = cutils_get_laser_end(
            lidar_range, perceive_distance, laser_index, heading_theta, vehicle_position_x,
            vehicle_position_y
        )
        laser_end = cutils_panda_position(point_x, point_y, height)
        result = physics_world.rayTestClosest(pg_start_position, laser_end, mask)
        node = result.getNode()
        hits = None
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
                break
        else:
            cloud_points[laser_index] = result.getHitFraction()
            if node:
                detected_objects.append(result)
                hits = result.hasHit()

        # update vis
        if require_colors:
            if hits:
                colors.append(cutils_add_cloud_point_vis(
                    result.getHitPos()[0], result.getHitPos()[1], result.getHitPos()[2], num_lasers, laser_index,
                    ANGLE_FACTOR,
                    MARK_COLOR0, MARK_COLOR1, MARK_COLOR2
                ))
            else:
                colors.append(cutils_add_cloud_point_vis(
                    laser_end[0], laser_end[1], height, num_lasers, laser_index, ANGLE_FACTOR, MARK_COLOR0, MARK_COLOR1,
                    MARK_COLOR2
                ))

    return cloud_points, detected_objects, colors

@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def cutils_norm(np_float64_t x1, np_float64_t x2):
    return sqrt(x1 * x1 + x2 * x2)

@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def cutils_clip(np_float64_t a, np_float64_t low, np_float64_t high):
    return fmin(fmax(a, low), high)
