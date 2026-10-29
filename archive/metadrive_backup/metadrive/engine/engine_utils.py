import logging
from typing import Optional

from metadrive.engine.base_engine import BaseEngine
from metadrive.engine.core.engine_core import EngineCore


def _fix_offscreen_rendering():
    """A little workaround to fix issues in offscreen rendering.

    See: https://github.com/metadriverse/metadrive/issues/632
    """
    from metadrive.envs.base_env import BASE_DEFAULT_CONFIG
    from metadrive.engine.engine_utils import initialize_engine, close_engine
    config = BASE_DEFAULT_CONFIG.copy()
    config["debug"] = True
    initialize_engine(config)
    close_engine()


def initialize_engine(env_global_config):
    """
    Initialize the engine core. Each process should only have at most one instance of the engine.

    Args:
        env_global_config: the global config.

    Returns:
        The engine.
    """
    # As a workaround, address the potential bug when rendering in headless machine.
    if env_global_config["use_render"] is False and env_global_config["image_observation"] is True:
        _fix_offscreen_rendering()

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
