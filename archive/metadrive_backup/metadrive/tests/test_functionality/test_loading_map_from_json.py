import json

from metadrive import MetaDriveEnv
from metadrive.utils import recursive_equal
from metadrive.utils.pg.generate_maps import generate_maps


def _test_loaded_map_alignment():
    raise DeprecationWarning("We do not generate maps from file now!")
    # Generate the second round
    # for seed in [0, 1, 2, 100, 200, 300, 9999]:
    for seed in [0, 1, 2, 99]:
        env_config = {"start_seed": seed, "num_scenarios": 1}
        generate_maps(MetaDriveEnv, env_config.copy(), json_file_path="seed{}_v1.json".format(seed))
        # generate_maps(MetaDriveEnv, env_config.copy(), json_file_path="seed{}_v2.json".format(seed))
        with open("seed{}_v1.json".format(seed), "r") as f:
            saved_v1 = json.load(f)
        # with open("seed{}_v2.json".format(seed), "r") as f:
        #     saved_v2 = json.load(f)

        e = MetaDriveEnv(env_config.copy())
        e.reset()
        map_data_realtime_load = e.current_map.get_meta_data()
        map_data_in_json = saved_v1["map_data"][str(seed)]
        e.close()

        e = MetaDriveEnv({"start_seed": seed, "num_scenarios": 1})
        e.reset()
        map_data_realtime_generate = e.current_map.get_meta_data()
        e.close()

        e = MetaDriveEnv({"start_seed": seed, "num_scenarios": 10})
        e.reset(seed=seed)
        map_data_realtime_generate_in_multiple_maps = e.current_map.get_meta_data()
        e.close()

        # Assert v1 and v2 have same maps
        # recursive_equal(saved_v1, saved_v2, True)

        # Assert json and loaded maps are same
        recursive_equal(map_data_in_json, map_data_realtime_load, True)

        # Assert json and generated maps are same
        recursive_equal(map_data_in_json, map_data_realtime_generate, True)

        # Assert json and generated maps in
        recursive_equal(map_data_in_json, map_data_realtime_generate_in_multiple_maps, True)

    # print(saved_v1)


def test_map_buffering():
    env_config = {"num_scenarios": 5}
    e = MetaDriveEnv(env_config)
    try:
        for i in range(10):
            e.reset()
        assert any([v is not None for v in e.engine.map_manager.maps.values()])
    finally:
        e.close()


if __name__ == '__main__':
    # test_loaded_map_alignment()
    test_map_buffering()
