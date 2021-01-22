import typing as tp
from collections import namedtuple

from pgdrive.pg_config.space import Dict, Box, Discrete

PGBoxSpace = namedtuple("PGBoxSpace", "max min")
PGDiscreteSpace = namedtuple("PGDiscreteSpace", "number")
PGConstantSpace = namedtuple("PGConstantSpace", "value")


class PGSpace(Dict):
    """
    length = PGSpace(name="length",max=50.0,min=10.0)
    Usage:
    PGSpace({"lane_length":length})
    """
    def __init__(self, our_config: tp.Dict[str, tp.Union[PGBoxSpace, PGDiscreteSpace, PGConstantSpace]]):
        super(PGSpace, self).__init__(PGSpace.wrap2gym_space(our_config))
        self.parameters = set(our_config.keys())

    @staticmethod
    def wrap2gym_space(our_config):
        ret = dict()
        for key, value in our_config.items():
            if isinstance(value, PGBoxSpace):
                ret[key] = Box(low=value.min, high=value.max, shape=(1, ))
            elif isinstance(value, PGDiscreteSpace):
                ret[key] = Discrete(value.number)
            elif isinstance(value, PGConstantSpace):
                ret[key] = Box(low=value.value, high=value.value, shape=(1, ))
            else:
                raise ValueError("{} can not be wrapped in gym space".format(key))
        return ret


if __name__ == "__main__":
    """
    Test
    """
    config = {
        "length": PGBoxSpace(min=10.0, max=80.0),
        "angle": PGBoxSpace(min=50.0, max=360.0),
        "goal": PGDiscreteSpace(number=3)
    }
    config = PGSpace(config)
    print(config.sample())
    config.seed(1)
    print(config.sample())
    print(config.sample())
    config.seed(1)
    print(*config.sample()["length"])
