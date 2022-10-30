import pickle
import tqdm
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils import recursive_equal, setup_logger


def test_gen_map_read():
    env_num = 3
    generate_config = {"environment_num": env_num, "start_seed": 0}
    restore_config = {"environment_num": env_num, "start_seed": 0}

    setup_logger(debug=True)
    try:
        env = MetaDriveEnv(generate_config)
        env.reset()
        data = env.engine.map_manager.dump_all_maps(file_name="test_10maps.pickle")
        env.close()

        # Check load
        with open("test_10maps.pickle", "rb+") as f:
            restored_data = pickle.load(f)

        env = MetaDriveEnv(restore_config)
        env.reset()
        env.engine.map_manager.load_all_maps("test_10maps.pickle")

        for i in range(env_num):
            m = env.maps[i + restore_config["start_seed"]].get_meta_data()
            origin = restored_data[i]
            m["map_config"].pop("config")
            m["map_config"].pop("type")

            origin["map_config"].pop("config")
            origin["map_config"].pop("type")

            recursive_equal(m, origin, need_assert=True)
        for seed in tqdm.tqdm(range(env_num), desc="Test Scenario"):
            env.reset(force_seed=seed)
            for i in range(10):
                env.step(env.action_space.sample())
        print("Finish!")
    finally:
        env.close()


if __name__ == "__main__":
    test_gen_map_read()
