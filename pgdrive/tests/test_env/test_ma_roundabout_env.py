from pgdrive.envs.marl_envs.marl_inout_roundabout import MultiAgentRoundaboutEnv


def _act(env, action):
    assert env.action_space.contains(action)
    obs, reward, done, info = env.step(action)
    assert set(obs.keys()) == set(reward.keys()) == set(env.observation_space.spaces.keys())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, dict)
    assert isinstance(info, dict)
    assert isinstance(done, dict)
    return obs, reward, done, info


def test_ma_roundabout_env():
    env = MultiAgentRoundaboutEnv({"num_agents": 1, "vehicle_config": {"lidar": {"num_others": 8}}})
    try:
        obs = env.reset()
        assert env.observation_space.contains(obs)
        for step in range(100):
            act = {k: [1, 1] for k in env.vehicles.keys()}
            assert len(act) == 1
            o, r, d, i = _act(env, act)
            if step == 0:
                assert not any(d.values())
    finally:
        env.close()

    env = MultiAgentRoundaboutEnv({"num_agents": 1, "vehicle_config": {"lidar": {"num_others": 0}}})
    try:
        obs = env.reset()
        assert env.observation_space.contains(obs)
        for step in range(100):
            act = {k: [1, 1] for k in env.vehicles.keys()}
            assert len(act) == 1
            o, r, d, i = _act(env, act)
            if step == 0:
                assert not any(d.values())
    finally:
        env.close()

    env = MultiAgentRoundaboutEnv({"num_agents": 4, "vehicle_config": {"lidar": {"num_others": 8}}})
    try:
        obs = env.reset()
        assert env.observation_space.contains(obs)
        for step in range(100):
            act = {k: [1, 1] for k in env.vehicles.keys()}
            o, r, d, i = _act(env, act)
            if step == 0:
                assert not any(d.values())
    finally:
        env.close()

    env = MultiAgentRoundaboutEnv({"num_agents": 4, "vehicle_config": {"lidar": {"num_others": 0}}})
    try:
        obs = env.reset()
        assert env.observation_space.contains(obs)
        for step in range(100):
            act = {k: [1, 1] for k in env.vehicles.keys()}
            o, r, d, i = _act(env, act)
            if step == 0:
                assert not any(d.values())
    finally:
        env.close()


def test_ma_roundabout_horizon():
    # test horizon
    env = MultiAgentRoundaboutEnv({"horizon": 100, "num_agents": 4, "vehicle_config": {"lidar": {"num_others": 2}}})
    try:
        obs = env.reset()
        assert env.observation_space.contains(obs)
        last_keys = set(env.vehicles.keys())
        for step in range(1000):
            act = {k: [1, 1] for k in env.vehicles.keys()}
            o, r, d, i = _act(env, act)
            new_keys = set(env.vehicles.keys())
            if step == 0:
                assert not any(d.values())
            if any(d.values()):
                assert len(last_keys) <= 4  # num of agents
                assert len(new_keys) <= 4  # num of agents
                for k in new_keys.difference(last_keys):
                    assert k in o
                    assert k in d
                print("Step {}, Done: {}".format(step, d))
            if d["__all__"]:
                break
            last_keys = new_keys
    finally:
        env.close()


def test_ma_roundabout_reset():
    env = MultiAgentRoundaboutEnv({"horizon": 50, "num_agents": 4})
    try:
        obs = env.reset()
        assert env.observation_space.contains(obs)
        for step in range(1000):
            act = {k: [1, 1] for k in env.vehicles.keys()}
            o, r, d, i = _act(env, act)
            if step == 0:
                assert not any(d.values())
            if d["__all__"]:
                obs = env.reset()
                assert env.observation_space.contains(obs)
                assert set(env.observation_space.spaces.keys()) == set(env.action_space.spaces.keys()) == \
                       set(env.observations.keys()) == set(obs.keys()) == \
                       set(env.config["target_vehicle_configs"].keys())
    finally:
        env.close()


def test_ma_roundabout_reward_sign():
    """
    If agent is simply moving forward without any steering, it will at least gain ~100 rewards, since we have a long
    straight road before coming into roundabout.
    However, some bugs cause the vehicles receive negative reward by doing this behavior!
    """
    class TestEnv(MultiAgentRoundaboutEnv):
        _reborn_count = 0

        def _reborn(self, dead_vehicle_id):
            v = self.vehicles.pop(dead_vehicle_id)
            v.prepare_step([0, -1])
            new_id = "agent{}".format(self._next_agent_id)
            self._next_agent_id += 1
            self.vehicles[new_id] = v  # Put it to new vehicle id.
            self.dones[new_id] = False  # Put it in the internal dead-tracking dict.
            new_born_place = self._safe_born_places[self._reborn_count]  # <<=== Here!
            print("Current reborn count ", self._reborn_count)
            new_born_place_config = new_born_place["config"]
            v.vehicle_config.update(new_born_place_config)
            v.reset(self.current_map)
            v.update_state()
            obs = self.observations[dead_vehicle_id]
            self.observations[new_id] = obs
            self.observations[new_id].reset(self, v)
            new_obs = self.observations[new_id].observe(v)
            self.observation_space.spaces[new_id] = self.observation_space.spaces[dead_vehicle_id]
            old_act_space = self.action_space.spaces.pop(dead_vehicle_id)
            self.action_space.spaces[new_id] = old_act_space
            return new_obs, new_id

    env = TestEnv({"num_agents": 1})
    try:
        obs = env.reset()
        ep_reward = 0.0
        for step in range(1000):
            act = {k: [0, 1] for k in env.vehicles.keys()}
            o, r, d, i = env.step(act)
            ep_reward += next(iter(r.values()))
            if any(d.values()):
                env._reborn_count += 1
                assert ep_reward > 50, ep_reward
                ep_reward = 0
            if env._reborn_count > len(env._safe_born_places):
                break
            if d["__all__"]:
                break
    finally:
        env.close()


if __name__ == '__main__':
    # test_ma_roundabout_env()
    # test_ma_roundabout_horizon()
    # test_ma_roundabout_reset()
    test_ma_roundabout_reward_sign()
