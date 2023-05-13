from gym.envs.registration import register, registry
from packaging import version
import gym
from metadrive.envs import MetaDriveEnv
from metadrive.envs import MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentRoundaboutEnv, \
    MultiAgentIntersectionEnv, MultiAgentParkingLotEnv, MultiAgentMetaDrive
from metadrive.envs import SafeMetaDriveEnv

metadrive_environment_dict = {
    "MetaDrive-validation-v0": {
        "start_seed": 0,
        "num_scenarios": 1000
    },
    "MetaDrive-10env-v0": {
        "start_seed": 1000,
        "num_scenarios": 10
    },
    "MetaDrive-100envs-v0": {
        "start_seed": 1000,
        "num_scenarios": 100
    },
    "MetaDrive-1000envs-v0": {
        "start_seed": 1000,
        "num_scenarios": 1000
    },
    # "MetaDrive-test-v0": {
    #     "start_seed": 0,
    #     "num_scenarios": 200
    # },
    # "MetaDrive-training0-v0": {
    #     "start_seed": 3000,
    #     "num_scenarios": 1000
    # },
    # "MetaDrive-training1-v0": {
    #     "start_seed": 5000,
    #     "num_scenarios": 1000
    # },
    # "MetaDrive-training2-v0": {
    #     "start_seed": 7000,
    #     "num_scenarios": 1000
    # },
}

safe_metadrive_environment_dict = {
    "SafeMetaDrive-validation-v0": {
        "start_seed": 0,
        "num_scenarios": 100
    },
    "SafeMetaDrive-10env-v0": {
        "start_seed": 1000,
        "num_scenarios": 10
    },
    "SafeMetaDrive-100envs-v0": {
        "start_seed": 1000,
        "num_scenarios": 100
    },
    "SafeMetaDrive-1000envs-v0": {
        "start_seed": 1000,
        "num_scenarios": 1000
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
existing_space = registry.env_specs if version.parse(gym.__version__) < version.parse("0.24.0") else registry
for env_name, env_config in metadrive_environment_dict.items():
    if env_name not in existing_space:
        envs.append(env_name)
        register(id=env_name, entry_point=MetaDriveEnv, kwargs=dict(config=env_config))

for env_name, env_config in safe_metadrive_environment_dict.items():
    if env_name not in existing_space:
        envs.append(env_name)
        register(id=env_name, entry_point=SafeMetaDriveEnv, kwargs=dict(config=env_config))

for env_name, entry in marl_env.items():
    if env_name not in existing_space:
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
