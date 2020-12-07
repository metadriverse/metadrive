import json
import os.path as osp

from pg_drive import GeneralizationRacing

root = osp.dirname(osp.dirname(osp.abspath(__file__)))
assert_path = osp.join(root, "assets", "maps")

predefined_maps = {
    "PG-Drive-maps": {
        "start_seed": 0,
        "environment_num": 10000
    },
}

if __name__ == '__main__':
    print("Root path is {}. Asset path is {}.".format(root, assert_path))
    for env_name, env_config in predefined_maps.items():
        env = GeneralizationRacing(env_config)
        data = env.dump_all_maps()
        file_path = osp.join(assert_path, "{}.json".format(env_name))
        with open(file_path, "w") as f:
            json.dump(data, f)
        env.close()
        print("Finish environment: ", env_name)

    # For test purpose only. Generate another group of maps with "-quanyi" suffix, and compare them
    #  with the original one.

    # # Generate the second round
    # for env_name, env_config in environment_set_dict.items():
    #     env = GeneralizationRacing(env_config)
    #     data = env.dump_all_maps()
    #     file_path = osp.join(assert_path, "{}-quanyi.json".format(env_name))
    #     with open(file_path, "w") as f:
    #         json.dump(data, f)
    #     env.close()
    #     print("Finish environment: ", env_name)
    #
    # from pg_drive.tests.generalization_env_test.test_gen_map_read import recursive_equal
    # for env_name, env_config in environment_set_dict.items():
    #     with open(osp.join(assert_path, "{}.json".format(env_name)), "r") as f:
    #         data_zhenghao = json.load(f)
    #     with open(osp.join(assert_path, "{}-quanyi2.json".format(env_name)), "r") as f:
    #         data_quanyi = json.load(f)
    #     recursive_equal(data_zhenghao, data_quanyi, True)
