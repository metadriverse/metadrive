"""
This filed is mostly copied from gym==0.17.2
We use the gym.spaces as helpers, but it may cause problem if user using some old version of gym.
"""

import logging
import typing as tp
from collections import namedtuple, OrderedDict

import numpy as np

from metadrive.utils import get_np_random

BoxSpace = namedtuple("BoxSpace", "max min")
DiscreteSpace = namedtuple("DiscreteSpace", "max min")
ConstantSpace = namedtuple("ConstantSpace", "value")


class Space:
    """
    Copied from gym: gym/spaces/space.py

    Defines the observation and action spaces, so you can write generic
    code that applies to any Env. For example, you can choose a random
    action.
    """
    def __init__(self, shape=None, dtype=None):
        import numpy as np  # takes about 300-400ms to import, so we load lazily
        self.shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else np.dtype(dtype)
        self.np_random = None
        self.seed()

    def sample(self):
        """Randomly sample an element of this space. Can be
        uniform or non-uniform sampling based on boundedness of space."""
        raise NotImplementedError

    def seed(self, seed=None):
        """Seed the PRNG of this space. """
        self.np_random, seed = get_np_random(seed, return_seed=True)
        return [seed]

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        raise NotImplementedError

    def __contains__(self, x):
        return self.contains(x)

    def to_jsonable(self, sample_n):
        """Convert a batch of samples from this space to a JSONable data type."""
        # By default, assume identity is JSONable
        return sample_n

    def from_jsonable(self, sample_n):
        """Convert a JSONable data type to a batch of samples from this space."""
        # By default, assume identity is JSONable
        return sample_n


class Dict(Space):
    """
    Copied from gym: gym/spaces/dcit.py

    A dictionary of simpler spaces.

    Example usage:
    self.observation_space = spaces.Dict({"position": spaces.Discrete(2), "velocity": spaces.Discrete(3)})

    Example usage [nested]:
    self.nested_observation_space = spaces.Dict({
        'sensors':  spaces.Dict({
            'position': spaces.Box(low=-100, high=100, shape=(3,)),
            'velocity': spaces.Box(low=-1, high=1, shape=(3,)),
            'front_cam': spaces.Tuple((
                spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                spaces.Box(low=0, high=1, shape=(10, 10, 3))
            )),
            'rear_cam': spaces.Box(low=0, high=1, shape=(10, 10, 3)),
        }),
        'ext_controller': spaces.MultiDiscrete((5, 2, 2)),
        'inner_state':spaces.Dict({
            'charge': spaces.Discrete(100),
            'system_checks': spaces.MultiBinary(10),
            'job_status': spaces.Dict({
                'task': spaces.Discrete(5),
                'progress': spaces.Box(low=0, high=100, shape=()),
            })
        })
    })
    """
    def __init__(self, spaces=None, **spaces_kwargs):
        assert (spaces is None) or (not spaces_kwargs), 'Use either Dict(spaces=dict(...)) or Dict(foo=x, bar=z)'
        if spaces is None:
            spaces = spaces_kwargs
        if isinstance(spaces, dict) and not isinstance(spaces, OrderedDict):
            spaces = OrderedDict(sorted(list(spaces.items())))
        if isinstance(spaces, list):
            spaces = OrderedDict(spaces)
        self.spaces = spaces
        for space in spaces.values():
            assert isinstance(space, Space), 'Values of the dict should be instances of gym.Space'
        super(Dict, self).__init__(None, None)  # None for shape and dtype, since it'll require special handling

    def seed(self, seed=None):
        [space.seed(seed) for space in self.spaces.values()]

    def sample(self):
        return OrderedDict([(k, space.sample()) for k, space in self.spaces.items()])

    def contains(self, x):
        if not isinstance(x, dict) or len(x) != len(self.spaces):
            return False
        for k, space in self.spaces.items():
            if k not in x:
                return False
            if not space.contains(x[k]):
                return False
        return True

    def __getitem__(self, key):
        return self.spaces[key]

    def __repr__(self):
        return "Dict(" + ", ".join([str(k) + ":" + str(s) for k, s in self.spaces.items()]) + ")"

    def to_jsonable(self, sample_n):
        # serialize as dict-repr of vectors
        return {key: space.to_jsonable([sample[key] for sample in sample_n]) \
                for key, space in self.spaces.items()}

    def from_jsonable(self, sample_n):
        dict_of_list = {}
        for key, space in self.spaces.items():
            dict_of_list[key] = space.from_jsonable(sample_n[key])
        ret = []
        for i, _ in enumerate(dict_of_list[key]):
            entry = {}
            for key, value in dict_of_list.items():
                entry[key] = value[i]
            ret.append(entry)
        return ret

    def __eq__(self, other):
        return isinstance(other, Dict) and self.spaces == other.spaces


class ParameterSpace(Dict):
    """
    length = PGSpace(name="length",max=50.0,min=10.0)
    Usage:
    PGSpace({"lane_length":length})
    """
    def __init__(self, our_config: tp.Dict[str, tp.Union[BoxSpace, DiscreteSpace, ConstantSpace]]):
        super(ParameterSpace, self).__init__(ParameterSpace.wrap2gym_space(our_config))
        self.parameters = set(our_config.keys())

    @staticmethod
    def wrap2gym_space(our_config):
        ret = dict()
        for key, value in our_config.items():
            if isinstance(value, BoxSpace):
                ret[key] = Box(low=value.min, high=value.max, shape=(1, ))
            elif isinstance(value, DiscreteSpace):
                ret[key] = Box(low=value.min, high=value.max, shape=(1, ), dtype=np.int64)
            elif isinstance(value, ConstantSpace):
                ret[key] = Box(low=value.value, high=value.value, shape=(1, ))
            else:
                raise ValueError("{} can not be wrapped in gym space".format(key))
        return ret


class Parameter:
    """
    Block parameters and vehicle parameters
    """
    # block
    length = "length"
    radius = "radius"
    angle = "angle"
    goal = "goal"
    dir = "dir"
    radius_inner = "inner_radius"  # only for roundabout use
    radius_exit = "exit_radius"
    exit_length = "exit_length"  # The length of the exit parts straight lane, for roundabout use only.
    t_intersection_type = "t_type"
    lane_num = "lane_num"
    change_lane_num = "change_lane_num"
    decrease_increase = "decrease_increase"
    one_side_vehicle_num = "one_side_vehicle_number"

    # vehicle
    # vehicle_length = "v_len"
    # vehicle_width = "v_width"
    vehicle_height = "v_height"
    front_tire_longitude = "f_tire_long"
    rear_tire_longitude = "r_tire_long"
    tire_lateral = "tire_lateral"
    tire_axis_height = "tire_axis_height"
    tire_radius = "tire_radius"
    mass = "mass"  # kg
    heading = "heading"
    # steering_max = "steering_max"
    # engine_force_max = "e_f_max"
    # brake_force_max = "b_f_max"
    # speed_max = "s_max"

    # vehicle visualization
    vehicle_vis_z = "vis_z"
    vehicle_vis_y = "vis_y"
    vehicle_vis_h = "vis_h"
    vehicle_vis_scale = "vis_scale"


class VehicleParameterSpace:
    STATIC_BASE_VEHICLE = dict(
        wheel_friction=ConstantSpace(0.9),
        max_engine_force=ConstantSpace(800),
        max_brake_force=ConstantSpace(150),
        max_steering=ConstantSpace(40),
        max_speed=ConstantSpace(80),
    )
    STATIC_DEFAULT_VEHICLE = STATIC_BASE_VEHICLE

    BASE_VEHICLE = dict(
        wheel_friction=ConstantSpace(0.9),
        max_engine_force=BoxSpace(750, 850),
        max_brake_force=BoxSpace(80, 180),
        max_steering=ConstantSpace(40),
        max_speed=ConstantSpace(80),
    )
    DEFAULT_VEHICLE = BASE_VEHICLE

    S_VEHICLE = dict(
        wheel_friction=ConstantSpace(0.9),
        max_engine_force=BoxSpace(350, 550),
        max_brake_force=BoxSpace(35, 80),
        max_steering=ConstantSpace(50),
        max_speed=ConstantSpace(80),
    )
    M_VEHICLE = dict(
        wheel_friction=ConstantSpace(0.75),
        max_engine_force=BoxSpace(650, 850),
        max_brake_force=BoxSpace(60, 150),
        max_steering=ConstantSpace(45),
        max_speed=ConstantSpace(80),
    )
    L_VEHICLE = dict(
        wheel_friction=ConstantSpace(0.8),
        max_engine_force=BoxSpace(450, 650),
        max_brake_force=BoxSpace(60, 120),
        max_steering=ConstantSpace(40),
        max_speed=ConstantSpace(80),
    )
    XL_VEHICLE = dict(
        wheel_friction=ConstantSpace(0.7),
        max_engine_force=BoxSpace(500, 700),
        max_brake_force=BoxSpace(50, 100),
        max_steering=ConstantSpace(35),
        max_speed=ConstantSpace(80),
    )


class BlockParameterSpace:
    """
    Make sure the range of curve parameters covers the parameter space of other blocks,
    otherwise, an error may happen in navigation info normalization
    """
    STRAIGHT = {Parameter.length: BoxSpace(min=40.0, max=80.0)}
    BIDIRECTION = {Parameter.length: BoxSpace(min=40.0, max=80.0)}

    CURVE = {
        Parameter.length: BoxSpace(min=40.0, max=80.0),
        Parameter.radius: BoxSpace(min=25.0, max=60.0),
        Parameter.angle: BoxSpace(min=45, max=135),
        Parameter.dir: DiscreteSpace(min=0, max=1)
    }
    INTERSECTION = {
        Parameter.radius: ConstantSpace(10),
        Parameter.change_lane_num: DiscreteSpace(min=0, max=1),  # 0, 1
        Parameter.decrease_increase: DiscreteSpace(min=0, max=1)  # 0, decrease, 1 increase
    }
    ROUNDABOUT = {
        # The radius of the
        Parameter.radius_exit: BoxSpace(min=5, max=15),
        Parameter.radius_inner: BoxSpace(min=15, max=45),
        Parameter.angle: ConstantSpace(60)
    }
    T_INTERSECTION = {
        Parameter.radius: ConstantSpace(10),
        Parameter.t_intersection_type: DiscreteSpace(min=0, max=2),  # 3 different t type for previous socket
        Parameter.change_lane_num: DiscreteSpace(min=0, max=1),  # 0,1
        Parameter.decrease_increase: DiscreteSpace(min=0, max=1)  # 0, decrease, 1 increase
    }
    RAMP_PARAMETER = {
        Parameter.length: BoxSpace(min=20, max=40)  # accelerate/decelerate part length
    }
    FORK_PARAMETER = {
        Parameter.length: BoxSpace(min=20, max=40),  # accelerate/decelerate part length
        Parameter.lane_num: DiscreteSpace(min=0, max=1)
    }
    BOTTLENECK_PARAMETER = {
        Parameter.length: BoxSpace(min=20, max=50),  # the length of straigh part
        Parameter.lane_num: DiscreteSpace(min=1, max=2),  # the lane num increased or descreased now 1-2
        "bottle_len": ConstantSpace(20),
        "solid_center_line": ConstantSpace(0)  # bool, turn on yellow line or not
    }
    TOLLGATE_PARAMETER = {
        Parameter.length: ConstantSpace(20),  # the length of straigh part
    }
    PARKING_LOT_PARAMETER = {
        Parameter.one_side_vehicle_num: DiscreteSpace(min=2, max=10),
        Parameter.radius: ConstantSpace(value=4),
        Parameter.length: ConstantSpace(value=8)
    }


class Discrete(Space):
    r"""
    Copied from gym: gym/spaces/discrete.py

    A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.

    Example::

        >>> Discrete(2)

    """
    def __init__(self, n):
        assert n >= 0
        self.n = n
        super(Discrete, self).__init__((), np.int64)

    def sample(self):
        return self.np_random.randint(self.n)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.char in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        return isinstance(other, Discrete) and self.n == other.n


class Box(Space):
    """
    Copied from gym: gym/spaces/box.py

    A (possibly unbounded) box in R^n. Specifically, a Box represents the
    Cartesian product of n closed intervals. Each interval has the form of one
    of [a, b], (-oo, b], [a, oo), or (-oo, oo).

    There are two common use cases:

    * Identical bound for each dimension::
        >>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        Box(3, 4)

    * Independent bound for each dimension::
        >>> Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
        Box(2,)

    """
    def __init__(self, low, high, shape=None, dtype=np.float32):
        assert dtype is not None, 'dtype must be explicitly provided. '
        self.dtype = np.dtype(dtype)

        # determine shape if it isn't provided directly
        if shape is not None:
            shape = tuple(shape)
            assert np.isscalar(low) or low.shape == shape, "low.shape doesn't match provided shape"
            assert np.isscalar(high) or high.shape == shape, "high.shape doesn't match provided shape"
        elif not np.isscalar(low):
            shape = low.shape
            assert np.isscalar(high) or high.shape == shape, "high.shape doesn't match low.shape"
        elif not np.isscalar(high):
            shape = high.shape
            assert np.isscalar(low) or low.shape == shape, "low.shape doesn't match high.shape"
        else:
            raise ValueError("shape must be provided or inferred from the shapes of low or high")

        if np.isscalar(low):
            low = np.full(shape, low, dtype=dtype)

        if np.isscalar(high):
            high = np.full(shape, high, dtype=dtype)

        self.shape = shape
        self.low = low
        self.high = high

        def _get_precision(dtype):
            if np.issubdtype(dtype, np.floating):
                return np.finfo(dtype).precision
            else:
                return np.inf

        low_precision = _get_precision(self.low.dtype)
        high_precision = _get_precision(self.high.dtype)
        dtype_precision = _get_precision(self.dtype)
        if min(low_precision, high_precision) > dtype_precision:
            logging.warning("Box bound precision lowered by casting to {}".format(self.dtype))
        self.low = self.low.astype(self.dtype)
        self.high = self.high.astype(self.dtype)

        # Boolean arrays which indicate the interval type for each coordinate
        self.bounded_below = -np.inf < self.low
        self.bounded_above = np.inf > self.high

        super(Box, self).__init__(self.shape, self.dtype)

    def is_bounded(self, manner="both"):
        below = np.all(self.bounded_below)
        above = np.all(self.bounded_above)
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError("manner is not in {'below', 'above', 'both'}")

    def sample(self):
        """
        Generates a single random sample inside of the Box.

        In creating a sample of the box, each coordinate is sampled according to
        the form of the interval:

        * [a, b] : uniform distribution
        * [a, oo) : shifted exponential distribution
        * (-oo, b] : shifted negative exponential distribution
        * (-oo, oo) : normal distribution
        """
        high = self.high if self.dtype.kind == 'f' \
            else self.high.astype('int64') + 1
        sample = np.empty(self.shape)

        # Masking arrays which classify the coordinates according to interval
        # type
        unbounded = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below & self.bounded_above
        low_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        # Vectorized sampling by interval type
        sample[unbounded] = self.np_random.normal(size=unbounded[unbounded].shape)

        sample[low_bounded] = self.np_random.exponential(size=low_bounded[low_bounded].shape) + self.low[low_bounded]

        sample[upp_bounded] = -self.np_random.exponential(size=upp_bounded[upp_bounded].shape) + self.high[upp_bounded]

        sample[bounded] = self.np_random.uniform(low=self.low[bounded], high=high[bounded], size=bounded[bounded].shape)
        if self.dtype.kind == 'i':
            sample = np.floor(sample)

        return sample.astype(self.dtype)

    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        return x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        return "Box" + str(self.shape)

    def __eq__(self, other):
        return isinstance(other, Box) and \
               (self.shape == other.shape) and \
               np.allclose(self.low, other.low) and \
               np.allclose(self.high, other.high)


if __name__ == "__main__":
    """
    Test
    """
    config = {
        "length": BoxSpace(min=10.0, max=80.0),
        "angle": BoxSpace(min=50.0, max=360.0),
        "goal": DiscreteSpace(min=0, max=2)
    }
    config = ParameterSpace(config)
    print(config.sample())
    config.seed(1)
    print(config.sample())
    print(config.sample())
    config.seed(1)
    print(*config.sample()["length"])
