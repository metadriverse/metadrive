import json
import os.path as osp

from metadrive import MetaDriveEnv

root = osp.dirname(osp.dirname(osp.abspath(__file__)))
asset_path = osp.join(root, "assets", "maps")

# The test script of this file is in metadrive/tests/test_functionality/test_loading_map_from_json.py


def generate_maps(env_class, env_config, json_file_path):
    raise DeprecationWarning("All procedural generated maps are load online now")

    env = env_class(env_config)
    data = env.dump_all_maps()
    with open(json_file_path, "w") as f:
        json.dump(data, f)
    env.close()
    print('Finished! Saved at: ', json_file_path)


if __name__ == '__main__':
    to_generate_map_config = {
        "start_seed": 0,
        "num_scenarios": 30000,
    }
    generate_maps(
        MetaDriveEnv, to_generate_map_config,
        osp.join(asset_path, "20210814_generated_maps_start_seed_0_num_scenarios_30000.json")
    )
