import copy

from metadrive.envs.marl_envs.marl_inout_roundabout import MultiAgentRoundaboutEnv


class ChangeNEnv(MultiAgentRoundaboutEnv):
    def __init__(self, config=None):
        self._raw_input_config = copy.deepcopy(config)
        super(ChangeNEnv, self).__init__(config)

    def close_and_reset_num_agents(self, num_agents):
        config = copy.deepcopy(self._raw_input_config)
        self.close()
        config["num_agents"] = num_agents
        super(ChangeNEnv, self).__init__(config)


def test_change_agent_num():
    e = ChangeNEnv({})
    try:
        for num_agents in [1, 5, 10, 30]:
            e.close_and_reset_num_agents(num_agents)
            e.reset()
            for _ in range(100):
                e.step(e.action_space.sample())
                # e.render("topdown")
            # print("Hi!!! We are in environment now! Current agents: ", len(e.vehicles.keys()))
            e.close()
    finally:
        e.close()


if __name__ == '__main__':
    test_change_agent_num()
