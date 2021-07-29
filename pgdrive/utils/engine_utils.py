import copy

import logging
from typing import Optional

from pgdrive.engine.base_engine import BaseEngine


def initialize_engine(env_global_config, agent_manager):
    cls = BaseEngine
    if cls.singleton is None:
        # assert cls.global_config is not None, "Set global config before initialization BaseEngine"
        cls.singleton = cls(env_global_config)
    else:
        raise PermissionError("There should be only one BaseEngine instance in one process")
    add_managers(agent_manager)


def add_managers(agent_manager):
    from pgdrive.manager.map_manager import MapManager
    from pgdrive.manager.object_manager import TrafficSignManager
    from pgdrive.manager.traffic_manager import TrafficManager
    from pgdrive.manager.policy_manager import PolicyManager

    engine = get_engine()
    # Add managers to BaseEngine, the order will determine the function implement order, e.g. reset(), step()
    engine.register_manager("agent_manager", agent_manager)
    engine.register_manager("map_manager", MapManager())
    engine.register_manager("object_manager", TrafficSignManager())
    engine.register_manager("traffic_manager", TrafficManager())
    engine.register_manager("policy_manager", PolicyManager())


def get_engine():
    return BaseEngine.singleton


def engine_initialized():
    return False if BaseEngine.singleton is None else True


def close_engine():
    if BaseEngine.singleton is not None:
        BaseEngine.singleton.close()
        BaseEngine.singleton = None


def get_global_config():
    engine = get_engine()
    return copy.copy(engine.global_config)


def set_global_random_seed(random_seed: Optional[int]):
    """
    Update the random seed and random engine
    All subclasses of RandomEngine will hold the same random engine, after calling this function
    :param random_seed: int, random seed
    """
    engine = get_engine()
    if engine is not None:
        engine.seed(random_seed)
    else:
        logging.warning("BaseEngine is not launched, fail to sync seed to engine!")
