import json

from pgdrive import PGDriveEnv, PGDriveEnvV2
from pgdrive.utils import recursive_equal
from pgdrive.utils.generate_maps import generate_maps


def test_loaded_map_alignment():
    # Generate the second round
    # for seed in [0, 1, 2, 100, 200, 300, 9999]:
    for seed in [0, 1, 2, 99]:
        env_config = {"start_seed": seed, "environment_num": 1}
        generate_maps(PGDriveEnv, env_config.copy(), json_file_path="seed{}_v1.json".format(seed))
        # generate_maps(PGDriveEnvV2, env_config.copy(), json_file_path="seed{}_v2.json".format(seed))
        with open("seed{}_v1.json".format(seed), "r") as f:
            saved_v1 = json.load(f)
        # with open("seed{}_v2.json".format(seed), "r") as f:
        #     saved_v2 = json.load(f)

        e = PGDriveEnv(env_config.copy())
        e.reset()
        assert e.engine.global_config["load_map_from_json"] is True, (
            "If this assertion fail, it means you are not using the predefined maps. Please check the read_all_"
            "maps_from_json function in map_manager.py"
        )
        map_data_realtime_load = e.current_map.save_map()
        map_data_in_json = saved_v1["map_data"][str(seed)]
        e.close()

        e = PGDriveEnv({"start_seed": seed, "environment_num": 1, "load_map_from_json": False})
        e.reset()
        map_data_realtime_generate = e.current_map.save_map()
        e.close()

        e = PGDriveEnv({"start_seed": seed, "environment_num": 10, "load_map_from_json": False})
        e.reset(force_seed=seed)
        map_data_realtime_generate_in_multiple_maps = e.current_map.save_map()
        e.close()

        # Assert v1 and v2 have same maps
        # recursive_equal(saved_v1, saved_v2, True)

        # Assert json and loaded maps are same
        recursive_equal(map_data_in_json, map_data_realtime_load, True)

        # Assert json and generated maps are same
        recursive_equal(map_data_in_json, map_data_realtime_generate, True)

        # Assert json and generated maps in
        recursive_equal(map_data_in_json, map_data_realtime_generate_in_multiple_maps, True)

    print(saved_v1)


def test_map_buffering():
    env_config = {"environment_num": 5}
    e = PGDriveEnv(env_config)
    try:
        for i in range(10):
            e.reset()
        assert any([v is not None for v in e.engine.map_manager.pg_maps.values()])
    finally:
        e.close()


if __name__ == '__main__':
    # test_loaded_map_alignment()
    test_map_buffering()
