import logging
from typing import Optional

from pgdrive.engine.pgdrive_engine import PGDriveEngine
import copy

from pgdrive.utils import get_np_random


def initialize_pgdrive_engine(env_global_config, agent_manager):
    cls = PGDriveEngine
    if cls.singleton is None:
        # assert cls.global_config is not None, "Set global config before initialization PGDriveEngine"
        cls.singleton = cls(env_global_config, agent_manager)
    else:
        raise PermissionError("There should be only one PGDriveEngine instance in one process")


def get_pgdrive_engine():
    return PGDriveEngine.singleton


def pgdrive_engine_initialized():
    return False if PGDriveEngine.singleton is None else True


def close_pgdrive_engine():
    if PGDriveEngine.singleton is not None:
        PGDriveEngine.singleton.close()
        PGDriveEngine.singleton = None


def get_global_config():
    engine = get_pgdrive_engine()
    return copy.copy(engine.global_config)


def set_global_random_seed(random_seed: Optional[int]):
    """
    Update the random seed and random engine
    All subclasses of RandomEngine will hold the same random engine, after calling this function
    :param random_seed: int, random seed
    """
    engine = get_pgdrive_engine()
    if engine is not None:
        engine.seed(random_seed)
    else:
        logging.warning("PGDriveEngine is not launched, fail to sync seed to engine!")
