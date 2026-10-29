import time
from metadrive.envs.marl_envs.multi_agent_metadrive import MULTI_AGENT_METADRIVE_DEFAULT_CONFIG

MULTI_AGENT_METADRIVE_DEFAULT_CONFIG["force_seed_spawn_manager"] = True
import numpy as np
from gymnasium.spaces import Box, Dict

from metadrive.envs.marl_envs.marl_tollgate import MultiAgentTollgateEnv
from metadrive.utils import distance_greater, norm


def _check_spaces_before_reset(env):
    a = set(env.config["agent_configs"].keys())
    b = set(env.observation_space.spaces.keys())
    c = set(env.action_space.spaces.keys())
    assert a == b == c
    _check_space(env)


def _check_spaces_after_reset(env, obs=None):
    a = set(env.config["agent_configs"].keys())
    b = set(env.observation_space.spaces.keys())
    assert a == b
    _check_shape(env)

    if obs:
        assert isinstance(obs, dict)
        assert set(obs.keys()) == a


def _check_shape(env):
    b = set(env.observation_space.spaces.keys())
    c = set(env.action_space.spaces.keys())
    d = set(env.agents.keys())
    e = set(env.engine.agents.keys())
    f = set([k for k in env.observation_space.spaces.keys() if not env.dones[k]])
    assert d == e == f, (b, c, d, e, f)
    assert c.issuperset(d)
    _check_space(env)


def _check_space(env):
    assert isinstance(env.action_space, Dict)
    assert isinstance(env.observation_space, Dict)
    o_shape = None
    for k, s in env.observation_space.spaces.items():
        assert isinstance(s, Box)
        if o_shape is None:
            o_shape = s.shape
        assert s.shape == o_shape
    a_shape = None
    for k, s in env.action_space.spaces.items():
        assert isinstance(s, Box)
        if a_shape is None:
            a_shape = s.shape
        assert s.shape == a_shape


def _act(env, action):
    assert env.action_space.contains(action)
    obs, reward, terminated, truncated, info = env.step(action)
    _check_shape(env)
    if not terminated["__all__"]:
        assert len(env.agents) > 0
    if not (set(obs.keys()) == set(reward.keys()) == set(env.observation_space.spaces.keys())):
        raise ValueError
    assert env.observation_space.contains(obs)
    assert isinstance(reward, dict)
    assert isinstance(info, dict)
    assert isinstance(terminated, dict)
    assert isinstance(truncated, dict)
    return obs, reward, terminated, truncated, info


def test_ma_toll_env():
    for env in [MultiAgentTollgateEnv({"delay_done": 0, "num_agents": 1, "vehicle_config": {"lidar": {"num_others": 8}}}
                                      ), MultiAgentTollgateEnv({"num_agents": 1, "delay_done": 0,
                                                                "vehicle_config": {"lidar": {"num_others": 0}}}),
                MultiAgentTollgateEnv({"num_agents": 4, "delay_done": 0,
                                       "vehicle_config": {"lidar": {"num_others": 8}}}),
                MultiAgentTollgateEnv({"num_agents": 4, "delay_done": 0,
                                       "vehicle_config": {"lidar": {"num_others": 0}}}),
                MultiAgentTollgateEnv({"num_agents": 8, "delay_done": 0,
                                       "vehicle_config": {"lidar": {"num_others": 0}}})]:
        try:
            _check_spaces_before_reset(env)
            obs, _ = env.reset()
            _check_spaces_after_reset(env, obs)
            assert env.observation_space.contains(obs)
            for step in range(100):
                act = {k: [1, 1] for k in env.agents.keys()}
                o, r, tm, tc, i = _act(env, act)
                if step == 0:
                    assert not any(tm.values())
                    assert not any(tc.values())

        finally:
            env.close()


def test_ma_toll_horizon():
    # test horizon
    for _ in range(10):  # This function is really easy to break, repeat multiple times!
        env = MultiAgentTollgateEnv(
            {
                "horizon": 100,
                "num_agents": 4,
                "vehicle_config": {
                    "lidar": {
                        "num_others": 2
                    }
                },
                "out_of_road_penalty": 777,
                "out_of_road_cost": 778,
                "crash_done": False
            }
        )
        try:
            _check_spaces_before_reset(env)
            obs, _ = env.reset()
            _check_spaces_after_reset(env, obs)
            assert env.observation_space.contains(obs)
            last_keys = set(env.agents.keys())
            for step in range(1, 1000):
                act = {k: [1, 1] for k in env.agents.keys()}
                o, r, tm, tc, i = _act(env, act)
                new_keys = set(env.agents.keys())
                if step == 0:
                    assert not any(tm.values())
                if any(tm.values()):
                    assert len(last_keys) <= 4  # num of agents
                    assert len(new_keys) <= 4  # num of agents
                    for k in new_keys.difference(last_keys):
                        assert k in o
                        assert k in tm
                    # print("Step {}, Done: {}".format(step, d))

                for kkk, rrr in r.items():
                    if rrr == -777:
                        assert tm[kkk]
                        assert i[kkk]["cost"] == 778
                        assert i[kkk]["out_of_road"]

                for kkk, iii in i.items():
                    if "out_of_road" in iii and (iii["out_of_road"] or iii["cost"] == 778):
                        assert tm[kkk]
                        assert i[kkk]["cost"] == 778
                        assert i[kkk]["out_of_road"]
                        # assert r[kkk] == -777

                if tm["__all__"]:
                    break
                last_keys = new_keys
        finally:
            env.close()


def test_ma_toll_reset():
    env = MultiAgentTollgateEnv({"horizon": 50, "num_agents": 4})
    try:
        _check_spaces_before_reset(env)
        obs, _ = env.reset()
        _check_spaces_after_reset(env, obs)
        assert env.observation_space.contains(obs)
        for step in range(1000):
            act = {k: [1, 1] for k in env.agents.keys()}
            o, r, tm, tc, i = _act(env, act)
            if step == 0:
                assert not any(tm.values())
                assert not any(tc.values())
            if tm["__all__"]:
                obs, _ = env.reset()
                assert env.observation_space.contains(obs)

                _check_spaces_after_reset(env, obs)
                assert set(env.observation_space.spaces.keys()) == set(env.action_space.spaces.keys()) == \
                       set(env.observations.keys()) == set(obs.keys()) == \
                       set(env.config["agent_configs"].keys())

                break
    finally:
        env.close()

    # Put vehicles to destination and then reset. This might cause error if agent is assigned destination BEFORE reset.
    env = MultiAgentTollgateEnv({"horizon": 100, "num_agents": 32, "success_reward": 777})
    try:
        _check_spaces_before_reset(env)
        success_count = 0
        agent_count = 0
        obs, _ = env.reset()
        _check_spaces_after_reset(env, obs)
        assert env.observation_space.contains(obs)

        for num_reset in range(5):
            for step in range(1000):

                # for _ in range(2):
                #     act = {k: [1, 1] for k in env.agents.keys()}
                #     o, r, tm, tc, i = _act(env, act)

                # Force vehicle to success!
                for v_id, v in env.agents.items():
                    loc = v.navigation.final_lane.end
                    # vehicle will stack together to explode!
                    v.set_position(loc, height=int(v_id[5:]) * 2)
                    v.set_position(loc)
                    pos = v.position
                    np.testing.assert_almost_equal(pos, loc, decimal=3)
                    new_loc = v.navigation.final_lane.end
                    long, lat = v.navigation.final_lane.local_coordinates(v.position)
                    flag1 = (v.navigation.final_lane.length - 5 < long < v.navigation.final_lane.length + 5)
                    flag2 = (
                        v.navigation.get_current_lane_width() / 2 >= lat >=
                        (0.5 - v.navigation.get_current_lane_num()) * v.navigation.get_current_lane_width()
                    )
                    # if not env._is_arrive_destination(v):
                    # print('sss')
                    assert env._is_arrive_destination(v)

                act = {k: [0, 0] for k in env.agents.keys()}
                o, r, tm, tc, i = _act(env, act)

                for v in env.agents.values():
                    assert len(v.navigation.checkpoints) > 2

                for kkk, iii in i.items():
                    if "arrive_dest" in iii and iii["arrive_dest"]:
                        # # print("{} success!".format(kkk))
                        success_count += 1

                for kkk, ddd in tm.items():
                    if ddd and kkk != "__all__":
                        assert i[kkk]["arrive_dest"]
                        agent_count += 1

                for kkk, rrr in r.items():
                    if tm[kkk]:
                        assert rrr == 777

                if tm["__all__"]:
                    # print("Finish {} agents. Success {} agents.".format(agent_count, success_count))
                    o, _ = env.reset()
                    assert env.observation_space.contains(o)
                    _check_spaces_after_reset(env, o)
                    break
    finally:
        env.close()


def test_ma_toll_close_spawn():
    def _no_close_spawn(vehicles):
        vehicles = list(vehicles.values())
        for c1, v1 in enumerate(vehicles):
            for c2 in range(c1 + 1, len(vehicles)):
                v2 = vehicles[c2]
                dis = norm(v1.position[0] - v2.position[0], v1.position[1] - v2.position[1])
                assert distance_greater(v1.position, v2.position, length=2.2)

    MultiAgentTollgateEnv._DEBUG_RANDOM_SEED = 1
    env = MultiAgentTollgateEnv(
        {
            # "use_render": True,
            "horizon": 50,
            "num_agents": 12,
            "map_config": {
                "exit_length": 30
            }
        }
    )
    env.seed(100)
    try:
        _check_spaces_before_reset(env)
        for num_r in range(10):
            obs, _ = env.reset()
            _check_spaces_after_reset(env)
            for _ in range(10):
                o, r, tm, tc, i = env.step({k: [0, 0] for k in env.agents.keys()})
                assert not any(tm.values())
            _no_close_spawn(env.agents)
            # print('Finish {} resets.'.format(num_r))
    finally:
        env.close()
        MultiAgentTollgateEnv._DEBUG_RANDOM_SEED = None


def test_ma_toll_reward_done_alignment_1():
    # out of road
    env = MultiAgentTollgateEnv({"horizon": 200, "num_agents": 4, "out_of_road_penalty": 777, "crash_done": False})
    try:
        _check_spaces_before_reset(env)
        obs, _ = env.reset()
        _check_spaces_after_reset(env, obs)
        assert env.observation_space.contains(obs)
        for action in [-1, 1]:
            for step in range(5000):
                act = {k: [action, 1] for k in env.agents.keys()}
                o, r, tm, tc, i = _act(env, act)
                for kkk, ddd in tm.items():
                    if ddd and kkk != "__all__" and not tm["__all__"] and not i[kkk]["max_step"]:
                        if r[kkk] != -777:
                            raise ValueError
                        # assert r[kkk] == -777
                        assert i[kkk]["out_of_road"]
                        # # print('{} done passed!'.format(kkk))
                for kkk, rrr in r.items():
                    if rrr == -777:
                        assert tm[kkk]
                        assert i[kkk]["out_of_road"]
                        # # print('{} reward passed!'.format(kkk))
                if tm["__all__"]:
                    env.reset()
                    break
    finally:
        env.close()

    # crash
    env = MultiAgentTollgateEnv(
        {
            "horizon": 100,
            "num_agents": 2,
            "crash_vehicle_penalty": 1.7777,
            "crash_done": True,
            "delay_done": 0,

            # "use_render": True,
            #
            "top_down_camera_initial_z": 160
        }
    )
    # Force the seed here so that the agent1 and agent2 are in same heading! Otherwise they might be in vertical
    # heading and cause one of the vehicle raise "out of road" error!
    env._DEBUG_RANDOM_SEED = 1
    try:
        _check_spaces_before_reset(env)
        obs, _ = env.reset()
        _check_spaces_after_reset(env, obs)
        for step in range(5):
            act = {k: [0, 0] for k in env.agents.keys()}
            o, r, tm, tc, i = _act(env, act)
        env.agents["agent0"].set_position(env.agents["agent1"].position, height=1.2)
        for step in range(5000):
            act = {k: [0, 0] for k in env.agents.keys()}
            o, r, tm, tc, i = _act(env, act)

            if not any(tm.values()):
                continue

            assert sum(tm.values()) == 2

            for kkk in ['agent0', 'agent1']:
                iii = i[kkk]
                assert iii["crash_vehicle"]
                assert iii["crash"]
                # assert r[kkk] == -1.7777
                # for kkk, ddd in te.items():
                ddd = tm[kkk]
                if ddd and kkk != "__all__":
                    # assert r[kkk] == -1.7777
                    assert i[kkk]["crash_vehicle"]
                    assert i[kkk]["crash"]
                    # # print('{} done passed!'.format(kkk))
                # for kkk, rrr in r.items():
                rrr = r[kkk]
                if rrr == -1.7777:
                    assert tm[kkk]
                    assert i[kkk]["crash_vehicle"]
                    assert i[kkk]["crash"]
                    # # print('{} reward passed!'.format(kkk))
            # assert te["__all__"]
            # if te["__all__"]:
            break
    finally:
        env._DEBUG_RANDOM_SEED = None
        env.close()

    # crash with real fixed vehicle


def test_ma_toll_reward_done_alignment_2():
    # crash 2
    env = MultiAgentTollgateEnv(
        {
            "map_config": {
                "exit_length": 110,
                "lane_num": 1
            },
            # "use_render": True,
            #
            "horizon": 200,
            "num_agents": 24,
            "crash_vehicle_penalty": 1.7777,
            "crash_done": False
        }
    )
    try:
        _check_spaces_before_reset(env)
        obs, _ = env.reset()
        _check_spaces_after_reset(env, obs)
        for step in range(1):
            act = {k: [0, 0] for k in env.agents.keys()}
            o, r, tm, tc, i = _act(env, act)

        for v_id, v in env.agents.items():
            if v_id != "agent0":
                v.set_static(True)

        for step in range(5000):
            act = {k: [0, 1] for k in env.agents.keys()}
            o, r, tm, tc, i = _act(env, act)
            for kkk, iii in i.items():
                if iii["crash_vehicle"]:
                    assert iii["crash"]
                    # #assert r[kkk] == -1.7777
            for kkk, ddd in tm.items():
                if ddd and kkk != "__all__" and not tm["__all__"]:
                    assert i[kkk]["out_of_road"] or i[kkk]["arrive_dest"] or i[kkk]["crash_building"]
                    # # print('{} done passed!'.format(kkk))
            for kkk, rrr in r.items():
                if rrr == -1.7777:
                    # assert d[kkk]
                    assert i[kkk]["crash_vehicle"]
                    assert i[kkk]["crash"]
                    # # print('{} reward passed!'.format(kkk))
            if tm["agent0"]:
                break
            if tm["__all__"]:
                break
    finally:
        env.close()

    # success
    env = MultiAgentTollgateEnv(
        {
            "horizon": 100,
            "num_agents": 2,
            "success_reward": 999,
            "out_of_road_penalty": 555,
            "crash_done": True
        }
    )
    try:
        _check_spaces_before_reset(env)
        obs, _ = env.reset()
        _check_spaces_after_reset(env)
        env.agents["agent0"].set_position(env.agents["agent0"].navigation.final_lane.end)
        assert env.observation_space.contains(obs)
        for step in range(5000):
            act = {k: [0, 0] for k in env.agents.keys()}
            o, r, tm, tc, i = _act(env, act)
            if tm["__all__"]:
                break
            kkk = "agent0"
            # assert r[kkk] == 999
            assert i[kkk]["arrive_dest"]
            assert tm[kkk]

            kkk = "agent1"
            # assert r[kkk] != 999
            assert not i[kkk]["arrive_dest"]
            assert not tm[kkk]
            break
    finally:
        env.close()


def test_ma_toll_reward_sign():
    """
    If agent is simply moving forward without any steering, it will at least gain ~100 rewards, since we have a long
    straight road before coming into toll.
    However, some bugs cause the vehicles receive negative reward by doing this behavior!
    """
    class TestEnv(MultiAgentTollgateEnv):
        _respawn_count = 0

        @property
        def _safe_places(self):
            safe_places = []
            for c, bid in enumerate(self.engine.spawn_manager.safe_spawn_places.keys()):
                safe_places.append((bid, self.engine.spawn_manager.safe_spawn_places[bid]))
            return safe_places

    env = TestEnv({"num_agents": 1})
    try:
        _check_spaces_before_reset(env)
        obs, _ = env.reset()
        _check_spaces_after_reset(env)
        ep_reward = 0.0
        for step in range(1000):
            act = {k: [0, 1] for k in env.agents.keys()}
            o, r, tm, tc, i = env.step(act)
            ep_reward += next(iter(r.values()))
            if any(tm.values()):
                # print("Finish respawn count: {}, reward {}".format(env._respawn_count, ep_reward))
                env._respawn_count += 1
                assert ep_reward > 10, ep_reward
                ep_reward = 0
            if env._respawn_count >= len(env._safe_places):
                break
            if tm["__all__"]:
                break
    finally:
        env.close()


def test_ma_toll_init_space():
    try:
        for start_seed in [5000, 6000, 7000]:
            for num_agents in [16, 32]:
                for num_others in [0, 2, 4, 8]:
                    for crash_vehicle_penalty in [0, 5]:
                        env_config = dict(
                            start_seed=start_seed,
                            num_agents=num_agents,
                            vehicle_config=dict(lidar=dict(num_others=num_others)),
                            crash_vehicle_penalty=crash_vehicle_penalty
                        )
                        env = MultiAgentTollgateEnv(env_config)

                        single_space = env.observation_space["agent0"]
                        assert single_space.shape is not None, single_space
                        assert np.prod(single_space.shape) is not None, single_space

                        single_space = env.action_space["agent0"]
                        assert single_space.shape is not None, single_space
                        assert np.prod(single_space.shape) is not None, single_space

                        _check_spaces_before_reset(env)
                        env.reset()
                        _check_spaces_after_reset(env)
                        env.close()
                        # print('Finish: ', env_config)
    finally:
        if "env" in locals():
            env.close()


def test_ma_toll_no_short_episode():
    env = MultiAgentTollgateEnv({
        "horizon": 300,
        "num_agents": 36,
    })
    try:
        _check_spaces_before_reset(env)
        o, _ = env.reset()
        _check_spaces_after_reset(env, o)
        actions = [[0, 1], [1, 1], [-1, 1]]
        start = time.time()
        tm_count = 0
        tm = {"__all__": False}
        for step in range(2000):
            # act = {k: actions[np.random.choice(len(actions))] for k in o.keys()}
            act = {k: actions[np.random.choice(len(actions))] for k in env.agents.keys()}
            o_keys = set(o.keys()).union({"__all__"})
            a_keys = set(env.action_space.spaces.keys()).union(set(tm.keys()))
            assert o_keys == a_keys
            o, r, tm, tc, i = _act(env, act)
            for kkk, iii in i.items():
                if tm[kkk]:
                    assert iii["episode_length"] >= 1
                    tm_count += 1
            if tm["__all__"]:
                o, _ = env.reset()
                tm = {"__all__": False}
            # if (step + 1) % 100 == 0:
            #     # print(
            #         "Finish {}/2000 simulation steps. Time elapse: {:.4f}. Average FPS: {:.4f}".format(
            #             step + 1,
            #             time.time() - start, (step + 1) / (time.time() - start)
            #         )
            #     )
            if tm_count > 200:
                break
    finally:
        env.close()


def test_ma_toll_horizon_termination(vis=False):
    """
    Instead of driving out of road, it may collide with other cars as well! so assertion may fail.
    """
    # test horizon
    env = MultiAgentTollgateEnv(
        {
            "horizon": 100,
            "num_agents": 8,
            "debug_static_world": True,
            "debug_physics_world": True,
            "use_render": vis,
            "debug": True,
            "crash_done": False,
            "log_level": 50,
        }
    )
    try:
        for _ in range(1):  # This function is really easy to break, repeat multiple times!
            _check_spaces_before_reset(env)
            obs, _ = env.reset()
            # env.engine.toggleDebug()
            _check_spaces_after_reset(env, obs)
            assert env.observation_space.contains(obs)
            should_respawn = set()
            special_agents = set(["agent0", "agent7"])
            for step in range(1, 10000):
                act = {k: [0, 0] for k in env.agents.keys()}
                for v_id in act.keys():
                    if v_id in special_agents:
                        act[v_id] = [1, 1]  # Add some randomness
                    else:
                        if v_id in env.agents:
                            env.agents[v_id].set_static(True)
                obs, r, tm, tc, i = _act(env, act)

                # if vis:
                #     env.render(mode="topdown")

                if step == 0 or step == 1:
                    assert not any(tm.values())
                    assert not any(tc.values())

                if should_respawn:
                    for kkk in should_respawn:
                        assert kkk not in obs, "It seems the max_step agents is not respawn!"
                        assert kkk not in r
                        assert kkk not in tm
                        assert kkk not in tc
                        assert kkk not in i
                    should_respawn.clear()

                for kkk, ddd in tm.items():
                    if ddd and kkk in special_agents:
                        assert i[kkk]["out_of_road"] or i[kkk]["crash_vehicle"]

                    if ddd and kkk == "__all__":
                        # print("Current: ", step)
                        continue
                    if ddd and kkk not in special_agents:
                        assert i[kkk]["max_step"]
                        assert not i[kkk]["out_of_road"]
                        assert not i[kkk]["crash"]
                        assert not i[kkk]["crash_vehicle"]
                        should_respawn.add(kkk)

                if tm["__all__"]:
                    obs, _ = env.reset()
                    should_respawn.clear()
                    break
    finally:
        env.close()


def test_ma_toll_40_agent_reset_after_respawn():
    def check_pos(vehicles):
        while vehicles:
            v_1 = vehicles[0]
            for v_2 in vehicles[1:]:
                v_1_pos = v_1.position
                v_2_pos = v_2.position
                assert norm(
                    v_1_pos[0] - v_2_pos[0], v_1_pos[1] - v_2_pos[1]
                ) > v_1.WIDTH / 2 + v_2.WIDTH / 2, "Vehicles overlap after reset()"
            assert not v_1.crash_vehicle, "Vehicles overlap after reset()"
            vehicles.remove(v_1)

    env = MultiAgentTollgateEnv({"horizon": 50, "num_agents": 36})
    try:
        _check_spaces_before_reset(env)
        obs, _ = env.reset()
        _check_spaces_after_reset(env, obs)
        assert env.observation_space.contains(obs)
        for step in range(50):
            env.reset()
            check_pos(list(env.agents.values()))
            for v_id in list(env.agents.keys())[:20]:
                env.agent_manager._finish(v_id)
            env.step({k: [1, 1] for k in env.agents.keys()})
            env.step({k: [1, 1] for k in env.agents.keys()})
            env.step({k: [1, 1] for k in env.agents.keys()})
    finally:
        env.close()


def test_ma_no_reset_error():
    # It is possible that many agents are populated in the same spawn place!
    def check_pos(vehicles):
        while vehicles:
            v_1 = vehicles[0]
            for v_2 in vehicles[1:]:
                v_1_pos = v_1.position
                v_2_pos = v_2.position
                assert norm(
                    v_1_pos[0] - v_2_pos[0], v_1_pos[1] - v_2_pos[1]
                ) > v_1.WIDTH / 2 + v_2.WIDTH / 2, "Vehicles overlap after reset()"
            if v_1.crash_vehicle:
                x = 1
                raise ValueError("Vehicles overlap after reset()")
            vehicles.remove(v_1)

    env = MultiAgentTollgateEnv({"horizon": 300, "num_agents": 36, "delay_done": 0, "use_render": False})
    try:
        _check_spaces_before_reset(env)
        obs, _ = env.reset()
        _check_spaces_after_reset(env, obs)
        assert env.observation_space.contains(obs)
        for step in range(50):
            check_pos(list(env.agents.values()))
            o, r, tm, tc, i = env.step({k: [0, 1] for k in env.agents.keys()})
            env.reset()
            if tm["__all__"]:
                break
    finally:
        env.close()


def test_randomize_spawn_place():
    last_pos = {}
    env = MultiAgentTollgateEnv({"num_agents": 4, "use_render": False, "force_seed_spawn_manager": False})
    try:
        obs, _ = env.reset()
        for step in range(100):
            act = {k: [1, 1] for k in env.agents.keys()}
            last_pos = {kkk: v.position for kkk, v in env.agents.items()}
            o, r, tm, tc, i = env.step(act)
            obs, _ = env.reset()
            new_pos = {kkk: v.position for kkk, v in env.agents.items()}
            for kkk, new_p in new_pos.items():
                assert not np.all(new_p == last_pos[kkk]), (new_p, last_pos[kkk], kkk)
    finally:
        env.close()


if __name__ == '__main__':
    # test_ma_toll_env()
    # test_ma_toll_horizon()
    # test_ma_toll_reset()
    # test_ma_toll_reward_done_alignment_1()
    # test_ma_toll_close_spawn()
    # test_ma_toll_reward_sign()
    # test_ma_toll_init_space()
    # test_ma_toll_no_short_episode()
    test_ma_toll_horizon_termination()
    # test_ma_toll_40_agent_reset_after_respawn()
    # test_ma_no_reset_error()
    # test_randomize_spawn_place()
