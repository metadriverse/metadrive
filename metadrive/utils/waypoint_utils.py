import numpy as np
def interpolate(waypoints, original_frequency, target_frequency):
    """
    Interpolates the given waypoints(in world coordinate) to match the target frequency.
    """
    final_waypoints = []
    start = np.array([0, 0])
    num_points = target_frequency // original_frequency + 1  #(10/2 =5) How many points to prepent to waypoints[i]
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


from scipy.interpolate import CubicSpline


def generate_smooth_spline(waypoints, num_points=100):
    """
    Generate smooth splines through the given waypoints.

    :param waypoints: List of (x, y) tuples representing waypoints.
    :param num_points: Number of points to sample along the spline.
    :return: x and y coordinates of the sampled points.
    """
    waypoints = np.array(waypoints)
    x = waypoints[:, 0]
    y = waypoints[:, 1]

    # Fit cubic splines
    cs_x = CubicSpline(np.arange(len(x)), x, bc_type='natural')
    cs_y = CubicSpline(np.arange(len(y)), y, bc_type='natural')

    # Generate new points
    t = np.linspace(0, len(waypoints) - 1, num_points)
    x_spline = cs_x(t)
    y_spline = cs_y(t)

    return x_spline, y_spline
