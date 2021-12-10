from panda3d.core import Vec3

from metadrive.utils.math_utils import Vector


# In MetaDrive, the direction of y axis is adverse to Panda3d. It is required to use these function to transform when sync
# the two coordinates.
# MetaDrive:
#                     ^ x
#                     |
#                     |
#                     |----------> y
#                    Ego
#
# Panda3d:
#                     ^ x
#                     |
#                     |
#         y <---------|
#                    Ego
# Note: the models loaded in Panda3d are facing to y axis, and thus -90' is required to make it face to x axis
def panda_position(position, z=0.0) -> Vec3:
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
    return Vec3(position[0], -position[1], z)


def metadrive_position(position: Vec3) -> "Vector":
    """
    Transform the position in Panda3d to MetaDrive world
    :param position: Vec3, position in Panda3d
    :return: 2d position
    """
    # return np.array([position[0], -position[1]])
    # return position[0], -position[1]
    return Vector((position[0], -position[1]))


def panda_heading(heading: float) -> float:
    """
    Transform the heading in MetaDrive to Panda3d
    :param heading: float, heading in MetaDrive (degree)
    :return: heading (degree)
    """
    return -heading


def metadrive_heading(heading: float) -> float:
    """
    Transform the heading in Panda3d to MetaDrive
    :param heading: float, heading in panda3d (degree)
    :return: heading (degree)
    """
    return -heading


def waymo_2_metadrive_position(position):
    return position[0], -position[1]


def waymo_2_metadrive_heading(heading):
    return -heading
