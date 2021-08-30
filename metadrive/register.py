from gym.envs.registration import register

from metadrive.envs import MetaDriveEnv

environment_dict = {
    "MetaDrive-test-v0": {
        "start_seed": 0,
        "environment_num": 200
    },
    "MetaDrive-validation-v0": {
        "start_seed": 200,
        "environment_num": 800
    },
    "MetaDrive-v0": {
        "start_seed": 1000,
        "environment_num": 100
    },
    "MetaDrive-10envs-v0": {
        "start_seed": 1000,
        "environment_num": 10
    },
    "MetaDrive-1000envs-v0": {
        "start_seed": 1000,
        "environment_num": 1000
    },
    "MetaDrive-training0-v0": {
        "start_seed": 3000,
        "environment_num": 1000
    },
    "MetaDrive-training1-v0": {
        "start_seed": 5000,
        "environment_num": 1000
    },
    "MetaDrive-training2-v0": {
        "start_seed": 7000,
        "environment_num": 1000
    },
}

for env_name, env_config in environment_dict.items():
    register(id=env_name, entry_point=MetaDriveEnv, kwargs=dict(config=env_config))


def get_env_list():
    return list(environment_dict.keys())


print("Successfully registered the following environments: {}.".format(get_env_list()))

if __name__ == '__main__':
    # Test purpose only
    import gym

    env = gym.make("MetaDrive-v0")
    env.reset()
