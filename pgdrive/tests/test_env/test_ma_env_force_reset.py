from pgdrive.envs.marl_envs.marl_inout_roundabout import MultiAgentRoundaboutEnv


def test_ma_env_force_reset():
    e = MultiAgentRoundaboutEnv({'num_agents': 1})
    e.reset()
    assert len(e.vehicles) == e.num_agents == len(e.config["target_vehicle_configs"]) == 1

    e.close_and_reset_num_agents(num_agents=2)
    e.reset()
    assert len(e.vehicles) == e.num_agents == len(e.config["target_vehicle_configs"]) == 2

    e.close_and_reset_num_agents(num_agents=5)
    e.reset()
    assert len(e.vehicles) == e.num_agents == len(e.config["target_vehicle_configs"]) == 5

    e.close()


if __name__ == '__main__':
    test_ma_env_force_reset()
