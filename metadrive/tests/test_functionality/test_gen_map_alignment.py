from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils import recursive_equal, setup_logger


def test_gen_map_alignment():
    """
    For visualization check
    """
    env_num = 2
    step = 1000
    generate_config = {"num_scenarios": env_num, "start_seed": 0, "use_render": False}

    setup_logger(debug=True)
    try:
        env = MetaDriveEnv(generate_config)
        env.reset()
        data_1 = env.engine.map_manager.dump_all_maps(file_name="test_10maps.pickle")
        env.close()
        env = None

        env = MetaDriveEnv(generate_config)
        for i in range(env_num):
            env.reset(seed=i)
        data_2 = env.engine.map_manager.dump_all_maps(file_name="test_10maps.pickle")
        recursive_equal(data_1.copy(), data_2.copy(), need_assert=True)

    finally:
        env.close()


if __name__ == "__main__":
    test_gen_map_alignment()
