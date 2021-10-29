from gym.envs.registration import register, registry

from metadrive.envs import MetaDriveEnv
from metadrive.envs import SafeMetaDriveEnv
from metadrive.envs import MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentRoundaboutEnv, \
    MultiAgentIntersectionEnv, MultiAgentParkingLotEnv, MultiAgentMetaDrive

metadrive_environment_dict = {
    "MetaDrive-validation-v0": {
        "start_seed": 0,
        "environment_num": 1000
    },
    "MetaDrive-10env-v0": {
        "start_seed": 1000,
        "environment_num": 10
    },
    "MetaDrive-100envs-v0": {
        "start_seed": 1000,
        "environment_num": 100
    },
    "MetaDrive-1000envs-v0": {
        "start_seed": 1000,
        "environment_num": 1000
    },
    # "MetaDrive-test-v0": {
    #     "start_seed": 0,
    #     "environment_num": 200
    # },
    # "MetaDrive-training0-v0": {
    #     "start_seed": 3000,
    #     "environment_num": 1000
    # },
    # "MetaDrive-training1-v0": {
    #     "start_seed": 5000,
    #     "environment_num": 1000
    # },
    # "MetaDrive-training2-v0": {
    #     "start_seed": 7000,
    #     "environment_num": 1000
    # },
}

safe_metadrive_environment_dict = {
    "SafeMetaDrive-validation-v0": {
        "start_seed": 0,
        "environment_num": 100
    },
    "SafeMetaDrive-10env-v0": {
        "start_seed": 1000,
        "environment_num": 10
    },
    "SafeMetaDrive-100envs-v0": {
        "start_seed": 1000,
        "environment_num": 100
    },
    "SafeMetaDrive-1000envs-v0": {
        "start_seed": 1000,
        "environment_num": 1000
    },
}

marl_env = {
    "MARLTollgate-v0": MultiAgentTollgateEnv,
    "MARLBottleneck-v0": MultiAgentBottleneckEnv,
    "MARLRoundabout-v0": MultiAgentRoundaboutEnv,
    "MARLIntersection-v0": MultiAgentIntersectionEnv,
    "MARLParkingLot-v0": MultiAgentParkingLotEnv,
    "MARLMetaDrive-v0": MultiAgentMetaDrive
}

envs = []
for env_name, env_config in metadrive_environment_dict.items():
    if env_name not in registry.env_specs:
        envs.append(env_name)
        register(id=env_name, entry_point=MetaDriveEnv, kwargs=dict(config=env_config))

for env_name, env_config in safe_metadrive_environment_dict.items():
    if env_name not in registry.env_specs:
        envs.append(env_name)
        register(id=env_name, entry_point=SafeMetaDriveEnv, kwargs=dict(config=env_config))

for env_name, entry in marl_env.items():
    if env_name not in registry.env_specs:
        envs.append(env_name)
        register(id=env_name, entry_point=entry, kwargs=dict(config={}))

if len(envs) > 0:
    print("Successfully registered the following environments: {}.".format(envs))

if __name__ == '__main__':
    # Test purpose only
    import gym

    env = gym.make("MetaDrive-validation-v0")
    env.reset()
    env.close()

    env = gym.make("SafeMetaDrive-validation-v0")
    env.reset()
    env.close()

    env = gym.make("MARLTollgate-v0")
    env.reset()
    env.close()
