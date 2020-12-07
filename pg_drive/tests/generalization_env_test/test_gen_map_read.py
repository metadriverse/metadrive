import json

from pg_drive.envs.generalization_racing import GeneralizationRacing
from pg_drive.utils import recursive_equal, setup_logger

setup_logger(debug=True)

if __name__ == "__main__":
    env = GeneralizationRacing({
        "environment_num": 10,
    })
    data = env.dump_all_maps()
    env.close()
    with open("test_10maps.json", "w") as f:
        json.dump(data, f)

    with open("test_10maps.json", "r") as f:
        restored_data = json.load(f)

    env = GeneralizationRacing({
        "environment_num": 10,
    })
    env.lazy_init()
    env.pg_world.clear_world()
    print("Start loading.")
    env.load_all_maps(restored_data)

    for i in range(10):
        m = env.maps[i].save_map()
        recursive_equal(m, data["map_data"][i], need_assert=True)
    print("Finish!")
    env.close()
