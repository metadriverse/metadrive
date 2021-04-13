import gym
from pgdrive.envs.marl_envs.marl_inout_roundabout import MultiAgentRoundaboutEnv
from pgdrive.utils import distance_greater, norm


def _act(env, action):
    assert env.action_space.contains(action)
    obs, reward, done, info = env.step(action)
    if not (set(obs.keys()) == set(reward.keys()) == set(env.observation_space.spaces.keys())):
        print('sss')
    assert set(obs.keys()) == set(reward.keys()) == set(env.observation_space.spaces.keys())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, dict)
    assert isinstance(info, dict)
    assert isinstance(done, dict)
    return obs, reward, done, info


def test_ma_roundabout_env():
    env = MultiAgentRoundaboutEnv({"num_agents": 1, "vehicle_config": {"lidar": {"num_others": 8}}})
    assert isinstance(env.observation_space, gym.spaces.Dict)
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

    env = MultiAgentRoundaboutEnv({"num_agents": 8, "vehicle_config": {"lidar": {"num_others": 0}}})
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
    env = MultiAgentRoundaboutEnv(
        {
            "horizon": 100,
            "num_agents": 4,
            "vehicle_config": {
                "lidar": {
                    "num_others": 2
                }
            },
            "out_of_road_penalty": 777,
            "crash_done": False
        }
    )
    try:
        obs = env.reset()
        assert env.observation_space.contains(obs)
        last_keys = set(env.vehicles.keys())
        for step in range(1, 1000):
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

            for kkk, rrr in r.items():
                if rrr == -777:
                    assert d[kkk]

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


def test_ma_roundabout_close_born():
    def _no_close_born(vehicles):
        vehicles = list(vehicles.values())
        for c1, v1 in enumerate(vehicles):
            for c2 in range(c1 + 1, len(vehicles)):
                v2 = vehicles[c2]
                dis = norm(v1.position[0] - v2.position[0], v1.position[1] - v2.position[1])
                assert distance_greater(v1.position, v2.position, length=2.2)

    MultiAgentRoundaboutEnv.EXIT_LENGTH = 20
    MultiAgentRoundaboutEnv._DEBUG_RANDOM_SEED = 1
    env = MultiAgentRoundaboutEnv({"horizon": 50, "num_agents": 32})
    env.seed(100)
    try:
        for num_r in range(10):
            obs = env.reset()
            for _ in range(20):
                o, r, d, i = env.step({k: [0, 0] for k in env.vehicles.keys()})
                assert not any(d.values())
            _no_close_born(env.vehicles)
            print('Finish {} resets.'.format(num_r))
    finally:
        env.close()


def test_ma_roundabout_reward_done_alignment():
    # out of road
    env = MultiAgentRoundaboutEnv({"horizon": 100, "num_agents": 4, "out_of_road_penalty": 777, "crash_done": False})
    try:
        obs = env.reset()
        assert env.observation_space.contains(obs)
        for action in [-1, 1]:
            for step in range(5000):
                act = {k: [action, 1] for k in env.vehicles.keys()}
                o, r, d, i = _act(env, act)
                for kkk, ddd in d.items():
                    if ddd and kkk != "__all__":
                        assert r[kkk] == -777
                        assert i[kkk]["out_of_road"]
                        # print('{} done passed!'.format(kkk))
                for kkk, rrr in r.items():
                    if rrr == -777:
                        assert d[kkk]
                        assert i[kkk]["out_of_road"]
                        # print('{} reward passed!'.format(kkk))
                if d["__all__"]:
                    env.reset()
                    break
    finally:
        env.close()

    env = MultiAgentRoundaboutEnv(
        {
            "horizon": 100,
            "num_agents": 2,
            "crash_vehicle_penalty": 1.7777,
            "crash_done": True
        }
    )
    try:
        obs = env.reset()
        env.vehicles["agent0"].reset(env.current_map, pos=env.vehicles["agent1"].position)
        assert env.observation_space.contains(obs)
        for step in range(5000):
            act = {k: [0, 0] for k in env.vehicles.keys()}
            o, r, d, i = _act(env, act)
            if d["__all__"]:
                break
            for kkk, ddd in d.items():
                if ddd and kkk != "__all__":
                    assert r[kkk] == -1.7777
                    assert i[kkk]["crash_vehicle"]
                    assert i[kkk]["crash"]
                    # print('{} done passed!'.format(kkk))
            for kkk, rrr in r.items():
                if rrr == -1.7777:
                    assert d[kkk]
                    assert i[kkk]["crash_vehicle"]
                    assert i[kkk]["crash"]
                    # print('{} reward passed!'.format(kkk))
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

        def _update_agent_pos_configs(self, config):
            config = super(TestEnv, self)._update_agent_pos_configs(config)
            safe_places = []
            for c, bid in enumerate(self._born_places_manager.safe_born_places.keys()):
                safe_places.append((bid, self._born_places_manager.safe_born_places[bid]))
            self._safe_places = safe_places
            return config

        def _replace_vehicles(self, v):
            v.prepare_step([0, -1])
            bp_index, new_born_place = self._safe_places[self._reborn_count]
            new_born_place_config = new_born_place["config"]
            v.vehicle_config.update(new_born_place_config)
            v.reset(self.current_map)
            v.update_state()
            return bp_index

    env = TestEnv({"num_agents": 1})
    try:
        obs = env.reset()
        ep_reward = 0.0
        for step in range(1000):
            act = {k: [0, 1] for k in env.vehicles.keys()}
            o, r, d, i = env.step(act)
            ep_reward += next(iter(r.values()))
            if any(d.values()):
                print("Finish reborn count: {}, reward {}".format(env._reborn_count, ep_reward))
                env._reborn_count += 1
                assert ep_reward > 10, ep_reward
                ep_reward = 0
            if env._reborn_count >= len(env._safe_places):
                break
            if d["__all__"]:
                break
    finally:
        env.close()


if __name__ == '__main__':
    # test_ma_roundabout_env()
    # test_ma_roundabout_horizon()
    # test_ma_roundabout_reset()
    # test_ma_roundabout_reward_done_alignment()
    # test_ma_roundabout_close_born()
    test_ma_roundabout_reward_sign()
