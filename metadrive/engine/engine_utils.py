import logging
from typing import Optional

from metadrive.engine.base_engine import BaseEngine
from metadrive.engine.core.engine_core import EngineCore


def initialize_engine(env_global_config):
    cls = BaseEngine
    if cls.singleton is None:
        # assert cls.global_config is not None, "Set global config before initialization BaseEngine"
        cls.singleton = cls(env_global_config)
    else:
        raise PermissionError("There should be only one BaseEngine instance in one process")
    return cls.singleton


def get_engine() -> BaseEngine:
    return BaseEngine.singleton


def get_object(object_name):
    return get_engine().get_objects([object_name])


def engine_initialized():
    return False if BaseEngine.singleton is None else True


def close_engine():
    if BaseEngine.singleton is not None:
        BaseEngine.singleton.close()
        BaseEngine.singleton = None


def get_global_config():
    return EngineCore.global_config


def initialize_global_config(global_config):
    """
    You can, of course, preset the engine config before launching the engine.
    """
    assert not engine_initialized(), "Can not call this API after engine initialization!"
    EngineCore.global_config = global_config


def set_global_random_seed(random_seed: Optional[int]):
    """
    Update the random seed and random engine
    All subclasses of Randomizable will hold the same random engine, after calling this function
    :param random_seed: int, random seed
    """
    engine = get_engine()
    if engine is not None:
        engine.seed(random_seed)
    else:
        logging.warning("BaseEngine is not launched, fail to sync seed to engine!")
