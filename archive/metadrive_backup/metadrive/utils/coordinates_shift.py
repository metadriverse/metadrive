import numpy as np
from panda3d.core import Vec3
from metadrive.utils.math import Vector
from metadrive.utils.math import wrap_to_pi

#
# Now all coordinates are the same and are all in right-handed
# MetaDrive (right handed):
#                     ^ x
#                     |
#                     |
#         y<----------|
#               EgoCar/Object/Pedestrian/Cyclist and so on
#
# Panda3d (right handed):
#                     ^ x
#                     |
#                     |
#         y <---------|
#                world origin
# Note: Camera is facing to y-axis:
#                         ^ x
#                         |
#                         |
#                  | \    |
#         y <------|  |---|
#                  | /    |
#                camera origin

# Note: the models loaded in Panda3d are facing to y axis, and thus -90' is required to make it face to x axis
# These APIs are still available for compatibility, but doesn't apply any operation to the input vector/heading


def panda_vector(position, z=0.0) -> Vec3:
    """
    Give a 2d or 3d position in MetaDrive, transform it to Panda3d world.
    If the position is a 2d array, height will be determined by the value of z.
    if the position is a 3d array, the value of z will be invalid.
    :param position: 2d or 3d position
    :param z: the height of the position, when position is a 2d array
    :return: position represented in Vec3
    """
    if len(position) == 3:
        z = position[2]
    return Vec3(position[0], position[1], z)


def metadrive_vector(position):
    """
    Transform the position in Panda3d to MetaDrive world
    :param position: Vec3, position in Panda3d
    :return: 2d position
    """
    # return np.array([position[0], -position[1]])
    # return position[0], -position[1]
    return Vector([position[0], position[1]])


def panda_heading(heading: float) -> float:
    """
    Transform the heading in MetaDrive to Panda3d
    :param heading: float, heading in MetaDrive (degree)
    :return: heading (degree)
    """
    # return -heading
    return heading


def metadrive_heading(heading: float) -> float:
    """
    Transform the heading in Panda3d to MetaDrive
    :param heading: float, heading in panda3d (degree)
    :return: heading (degree)
    """
    # return -heading
    return heading
