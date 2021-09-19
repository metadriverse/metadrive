import copy

from metadrive.envs.marl_envs.marl_inout_roundabout import MultiAgentRoundaboutEnv


def test_ma_env_force_reset():
    def close_and_reset_num_agents(env, num_agents, raw_input_config):
        config = copy.deepcopy(raw_input_config)
        env.close()
        config["num_agents"] = num_agents
        env.__init__(config)

    config = {'num_agents': 1}
    e = MultiAgentRoundaboutEnv(config)
    _raw_input_config = copy.deepcopy(config)
    e.reset()
    assert len(e.vehicles) == e.num_agents == len(e.config["target_vehicle_configs"]) == 1

    close_and_reset_num_agents(e, num_agents=2, raw_input_config=_raw_input_config)
    e.reset()
    assert len(e.vehicles) == e.num_agents == len(e.config["target_vehicle_configs"]) == 2

    close_and_reset_num_agents(e, num_agents=5, raw_input_config=_raw_input_config)
    e.reset()
    assert len(e.vehicles) == e.num_agents == len(e.config["target_vehicle_configs"]) == 5

    e.close()


if __name__ == '__main__':
    test_ma_env_force_reset()
