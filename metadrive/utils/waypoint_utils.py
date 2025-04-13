import numpy as np


def interpolate(waypoints, original_frequency, target_frequency):
    """
    Interpolates the given waypoints(in world coordinate) to match the target frequency.
    """
    final_waypoints = []
    start = np.array([0, 0])
    num_points = target_frequency // original_frequency + 1  # (10/2 =5) How many points to prepent to waypoints[i]
    for i in range(len(waypoints)):
        # uniformally sample points between the two waypoints
        sampled_points = np.linspace(start, waypoints[i], num_points)
        final_waypoints.extend(sampled_points[1:])  # skip the first point to avoid duplication
        start = waypoints[i]
    return np.array(final_waypoints)


def interpolate_headings(waypoints):
    """
    Interpolates the headings of the waypoints.
    return the yaw angle(in world coordinate), with positive value turning left and negative value turning right
    """
    # Calculate the headings based on the waypoints
    headings = np.arctan2(np.diff(waypoints[:, 1]), np.diff(waypoints[:, 0]))
    # Append the last heading to match the length of waypoints
    headings = np.append(headings, headings[-1])
    return headings


def interpolate_angular_velocities(headings, time_interval):
    """
    Interpolates the angular velocities based on the headings and time interval.
    return in rad/s(in world coordinate), with positive value turning left and negative value turning right
    """
    # Calculate the angular velocities
    angular_velocities = np.diff(headings) / time_interval
    # Append the last angular velocity to match the length of headings
    angular_velocities = np.append(angular_velocities, angular_velocities[-1])
    return angular_velocities


def interpolate_velocities(waypoints, dt):
    """
    Interpolates the velocities vector based on the waypoints and time interval.
    """
    diff = np.diff(waypoints, axis=0)
    velocitaies = diff / dt
    # Append the last velocity to match the length of waypoints
    velocitaies = np.append(velocitaies, velocitaies[-1].reshape(1, -1), axis=0)
    return velocitaies


def rotate(x, y, angle, z=None, assert_shape=True):
    """
    Rotate the coordinates (x, y) by the given angle in radians.
    """
    if assert_shape:
        assert angle.shape == x.shape == y.shape, (angle.shape, x.shape, y.shape)
        if z is not None:
            assert x.shape == z.shape
    other_x_trans = np.cos(angle) * x - np.sin(angle) * y
    other_y_trans = np.cos(angle) * y + np.sin(angle) * x
    if z is None:
        output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)
    else:
        output_coords = np.stack((other_x_trans, other_y_trans, z), axis=-1)
    return output_coords
