from pgdrive.engine.pgdrive_engine import PGDriveEngine


def initialize_pgdrive_engine(config, agent_manager):
    cls = PGDriveEngine
    if cls.singleton is None:
        cls.singleton = cls(config, agent_manager)
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
