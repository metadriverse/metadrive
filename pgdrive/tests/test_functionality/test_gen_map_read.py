import json

from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.utils import recursive_equal, setup_logger


def test_gen_map_read():
    setup_logger(debug=True)

    env = PGDriveEnv({"environment_num": 10, "load_map_from_json": False})
    try:
        data = env.dump_all_maps()
        with open("test_10maps.json", "w") as f:
            json.dump(data, f)

        with open("test_10maps.json", "r") as f:
            restored_data = json.load(f)
        env.close()

        env = PGDriveEnv({
            "environment_num": 10,
        })
        env.lazy_init()
        env.engine.clear_world()
        print("Start loading.")
        env.engine.map_manager.load_all_maps(restored_data)

        while any([v is None for v in env.maps.values()]):
            env.reset()

        for i in range(10):
            m = env.maps[i].save_map()
            recursive_equal(m, data["map_data"][i], need_assert=True)
        print("Finish!")
    finally:
        env.close()


if __name__ == "__main__":
    test_gen_map_read()
