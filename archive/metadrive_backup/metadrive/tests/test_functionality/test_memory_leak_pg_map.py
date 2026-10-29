import time

from metadrive.component.map.pg_map import PGMap
from metadrive.engine.engine_utils import initialize_engine, close_engine
from metadrive.envs import MetaDriveEnv

try:
    from reprlib import repr
except ImportError:
    pass

from metadrive.tests.test_functionality.test_memory_leak_engine import process_memory, total_size


def test_pg_map_memory_leak():

    try:
        default_config = MetaDriveEnv.default_config()

        # default_config["map_config"]["config"] = "SSS"
        # default_config["map_config"]["type"] = "block_sequence"

        default_config["map_config"]["config"] = 1

        # default_config["debug"] = True
        # default_config["debug_physics_world"] = True

        close_engine()
        engine = initialize_engine(default_config)

        ct = time.time()
        last_lm = cm = process_memory()
        last_mem = 0.0
        for t in range(50):
            lt = time.time()

            our_map = PGMap(default_config["map_config"])

            # our_map.blocks.clear()
            # our_map.blocks = []

            size = total_size(our_map)
            # print("map size: {:,}".format(size))
            del our_map

            # print(engine.physics_world.report_bodies())

            nlt = time.time()
            lm = process_memory()
            # print(
            #     "After {} Iters, Time {:.3f} Total Time {:.3f}, Memory Usage {:,} Memory Change {:,}".format(
            # t + 1, nlt - lt, nlt - ct, lm - cm, lm - last_lm
            #     )
            # )
            # last_lm = lm
            # if t > 100:
            #     assert abs((lm - cm) - last_mem) < 1024  # Memory should not have change > 1KB
            # last_mem = lm - cm

    finally:
        close_engine()

    assert lm - cm < 1024 * 1024 * 120, "We expect will cause 100M memory leak with 50 iterations."


if __name__ == "__main__":
    test_pg_map_memory_leak()
