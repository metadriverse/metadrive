# DiamondSquare.py
# Written by Hailey K Buckingham
# On github, buckinha/DiamondSquare
#

import random
import math
import numpy as np


def diamond_square(
    shape: (int, int),
    min_height: [float or int],
    max_height: [float or int],
    roughness: [float or int],
    random_seed=None,
    uint16=True,
    minus_mean=True
):
    """Runs a diamond square algorithm and returns an array (or list) with the landscape

        An important difference (possibly) between this, and other implementations of the
    diamond square algorithm is how I use the roughness parameter. For each "perturbation"
    I pull a random number from a uniform distribution between min_height and max_height.
    I then take the weighted average between that value, and the average value of the
    "neighbors", whether those be in the diamond or in the square step, as normal. The
    weights used for the weighted sum are (roughness) and (1-roughness) for the random
    number and the average, respectively, where roughness is a float that always falls
    between 0 and 1.
        The roughness value used in each iteration is based on the roughness parameter
    passed in, and is computed as follows:

        this_iteration_roughness = roughness**iteration_number

    where the first iteration has iteration_number = 0. The first roughness value
    actually used (in the very first diamond and square step) is roughness**0 = 1. Thus,
    the values for those first diamond and square step entries will be entirely random.
    This effectively means that I am seeding with A 3x3 grid of random values, rather
    than with just the four corners.
        As the process continues, the weight placed on the random number draw falls from
    the original value of 1, to roughness**1, to roughness**2, and so on, ultimately
    approaching 0. This means that the values of new cells will slowly shift from being
    purely random, to pure averages.


    OTHER NOTES:
    Internally, all heights are between 0 and 1, and are rescaled at the end.


    PARAMETERS
    ----------
    :param shape
        tuple of ints, (int, int): the shape of the resulting landscape

    :param min_height
        Int or Float: The minimum height allowed on the landscape

    :param max_height
        Int or Float: The maximum height allowed on the landscape

    :param roughness
        Float with value between 0 and 1, reflecting how bumpy the landscape should be.
        Values near 1 will result in landscapes that are extremely rough, and have almost no
        cell-to-cell smoothness. Values near zero will result in landscapes that are almost
        perfectly smooth.

        Values above 1.0 will be interpreted as 1.0
        Values below 0.0 will be interpreted as 0.0

    :param random_seed
        Any value. Defaults to None. If a value is given, the algorithm will use it to seed the random
        number generator, ensuring replicability.

    :param as_ndarray
        Bool: whether the landscape should be returned as a numpy array. If set
        to False, the method will return list of lists.

    :returns [list] or nd_array
    """

    # sanitize inputs
    if roughness > 1:
        roughness = 1.0
    if roughness < 0:
        roughness = 0.0

    working_shape, iterations = _get_working_shape_and_iterations(shape)

    # create the array
    diamond_square_array = np.full(working_shape, -1, dtype='float')

    # seed the random number generator
    random.seed(random_seed)

    # seed the corners
    diamond_square_array[0, 0] = random.uniform(0, 1)
    diamond_square_array[working_shape[0] - 1, 0] = random.uniform(0, 1)
    diamond_square_array[0, working_shape[1] - 1] = random.uniform(0, 1)
    diamond_square_array[working_shape[0] - 1, working_shape[1] - 1] = random.uniform(0, 1)

    # do the algorithm
    for i in range(iterations):
        r = math.pow(roughness, i)

        step_size = math.floor((working_shape[0] - 1) / math.pow(2, i))

        _diamond_step(diamond_square_array, step_size, r)
        _square_step(diamond_square_array, step_size, r)

    # rescale the array to fit the min and max heights specified
    diamond_square_array = min_height + (diamond_square_array * (max_height - min_height))

    # trim array, if needed
    final_array = diamond_square_array[:shape[0], :shape[1]]

    if minus_mean:
        final_array -= np.mean(final_array)

    if uint16:
        final_array = final_array.astype(np.uint16)

    return final_array


def _get_working_shape_and_iterations(requested_shape, max_power_of_two=13):
    """Returns the necessary size for a square grid which is usable in a DS algorithm.

    The Diamond Square algorithm requires a grid of size n x n where n = 2**x + 1, for any
    integer value of x greater than two. To accomodate a requested map size other than these
    dimensions, we simply create the next largest n x n grid which can entirely contain the
    requested size, and return a subsection of it.

    This method computes that size.

    PARAMETERS
    ----------
    requested_shape
        A 2D list-like object reflecting the size of grid that is ultimately desired.

    max_power_of_two
        an integer greater than 2, reflecting the maximum size grid that the algorithm can EVER
        attempt to make, even if the requested size is too big. This limits the algorithm to
        sizes that are manageable, unless the user really REALLY wants to have a bigger one.
        The maximum grid size will have an edge of size  (2**max_power_of_two + 1)

    RETURNS
    -------
    An integer of value n, as described above.
    """
    if max_power_of_two < 3:
        max_power_of_two = 3

    largest_edge = max(requested_shape)

    for power in range(1, max_power_of_two + 1):
        d = (2**power) + 1
        if largest_edge <= d:
            return (d, d), power

    # failsafe: no values in the dimensions array were allowed, so print a warning and return
    # the maximum size.
    d = 2**max_power_of_two + 1
    print("DiamondSquare Warning: Requested size was too large. Grid of size {0} returned" "".format(d))
    return (d, d), max_power_of_two


def _diamond_step(DS_array, step_size, roughness):
    """Does the diamond step for a given iteration.

    During the diamond step, the diagonally adjacent cells are filled:

    Value   None   Value   None   Value  ...

    None   FILLING  None  FILLING  None  ...

    Value   None   Value   None   Value  ...

    ...     ...     ...     ...    ...   ...

    So we'll step with increment step_size over BOTH axes

    """
    # calculate where all the diamond corners are (the ones we'll be filling)
    half_step = math.floor(step_size / 2)
    x_steps = range(half_step, DS_array.shape[0], step_size)
    y_steps = x_steps[:]

    for i in x_steps:
        for j in y_steps:
            if DS_array[i, j] == -1.0:
                DS_array[i, j] = _diamond_displace(DS_array, i, j, half_step, roughness)


def _square_step(DS_array, step_size, roughness):
    """Does the square step for a given iteration.

    During the diamond step, the diagonally adjacent cells are filled:

     Value    FILLING    Value    FILLING   Value   ...

    FILLING   DIAMOND   FILLING   DIAMOND  FILLING  ...

     Value    FILLING    Value    FILLING   Value   ...

      ...       ...       ...       ...      ...    ...

    So we'll step with increment step_size over BOTH axes

    """

    # doing this in two steps: the first, where the every other column is skipped
    # and the second, where every other row is skipped. For each, iterations along
    # the half-steps go vertically or horizontally, respectively.

    # set the half-step for the calls to square_displace
    half_step = math.floor(step_size / 2)

    # vertical step
    steps_x_vert = range(half_step, DS_array.shape[0], step_size)
    steps_y_vert = range(0, DS_array.shape[1], step_size)

    # horizontal step
    steps_x_horiz = range(0, DS_array.shape[0], step_size)
    steps_y_horiz = range(half_step, DS_array.shape[1], step_size)

    for i in steps_x_horiz:
        for j in steps_y_horiz:
            DS_array[i, j] = _square_displace(DS_array, i, j, half_step, roughness)

    for i in steps_x_vert:
        for j in steps_y_vert:
            DS_array[i, j] = _square_displace(DS_array, i, j, half_step, roughness)


def _diamond_displace(DS_array, i, j, half_step, roughness):
    """
    defines the midpoint displacement for the diamond step
    :param DS_array:
    :param i:
    :param j:
    :param half_step:
    :param roughness:
    :return:
    """
    ul = DS_array[i - half_step, j - half_step]
    ur = DS_array[i - half_step, j + half_step]
    ll = DS_array[i + half_step, j - half_step]
    lr = DS_array[i + half_step, j + half_step]

    ave = (ul + ur + ll + lr) / 4.0

    rand_val = random.uniform(0, 1)

    return (roughness * rand_val) + (1.0 - roughness) * ave


def _square_displace(DS_array, i, j, half_step, roughness):
    """
    Defines the midpoint displacement for the square step

    :param DS_array:
    :param i:
    :param j:
    :param half_step:
    :param roughness:
    :return:
    """
    _sum = 0.0
    divide_by = 4

    # check cell "above"
    if i - half_step >= 0:
        _sum += DS_array[i - half_step, j]
    else:
        divide_by -= 1

    # check cell "below"
    if i + half_step < DS_array.shape[0]:
        _sum += DS_array[i + half_step, j]
    else:
        divide_by -= 1

    # check cell "left"
    if j - half_step >= 0:
        _sum += DS_array[i, j - half_step]
    else:
        divide_by -= 1

    # check cell "right"
    if j + half_step < DS_array.shape[0]:
        _sum += DS_array[i, j + half_step]
    else:
        divide_by -= 1

    ave = _sum / divide_by

    rand_val = random.uniform(0, 1)

    return (roughness * rand_val) + (1.0 - roughness) * ave
