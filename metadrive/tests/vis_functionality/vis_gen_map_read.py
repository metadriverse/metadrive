import pickle

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils import setup_logger


def vis_gen_map_read():
    """
    For visualization check
    """
    env_num = 3
    step = 500
    generate_config = {"num_scenarios": env_num, "start_seed": 0, "use_render": True}
    restore_config = {"num_scenarios": env_num, "start_seed": 0, "use_render": True}

    setup_logger(debug=True)
    try:
        env = MetaDriveEnv(generate_config)
        env.reset()
        data = env.engine.map_manager.dump_all_maps(file_name="test_10maps.pickle")
        for i in range(step):
            env.step(env.action_space.sample())
        env.close()

        # Check load
        with open("test_10maps.pickle", "rb") as f:
            restored_data = pickle.load(f)

        env = MetaDriveEnv(restore_config)
        env.reset()
        env.engine.map_manager.load_all_maps("test_10maps.pickle")
        env.reset()

        for i in range(step):
            env.step(env.action_space.sample())
    finally:
        env.close()


if __name__ == "__main__":
    vis_gen_map_read()
