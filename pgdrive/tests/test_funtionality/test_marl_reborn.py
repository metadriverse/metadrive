from pgdrive.envs.marl_envs.marl_inout_roundabout import MultiAgentRoundaboutEnv
from pgdrive.utils import setup_logger


def test_respawn():
    out_of_road_cost = 5555
    out_of_road_penalty = 2222
    env = MultiAgentRoundaboutEnv(
        {
            "num_agents": 2,
            "out_of_road_cost": out_of_road_cost,
            "out_of_road_penalty": out_of_road_penalty,
            # "use_render": True, "fast": True
        }
    )

    assert set(env.observations.keys()) == {"agent0", "agent1"}
    assert set(env.action_space.spaces.keys()) == {"agent0", "agent1"}
    assert set(env.config.target_vehicle_configs.keys()) == {"agent0", "agent1"}
    assert set(env.vehicles.keys()) == set()  # Not initialized yet!

    o = env.reset()

    assert set(o.keys()) == {"agent0", "agent1"}
    assert set(env.observations.keys()) == {"agent0", "agent1"}
    assert set(env.action_space.spaces.keys()) == {"agent0", "agent1"}
    assert set(env.config.target_vehicle_configs.keys()) == {"agent0", "agent1"}
    assert set(env.vehicles.keys()) == {"agent0", "agent1"}

    v_id_0 = "agent0"
    v_id_1 = "agent1"
    count = 2
    tracks = []
    done_count = 0
    for i in range(1, 1000):
        o, r, d, info = env.step({v_id_0: [-1, 1], v_id_1: [1, 1]})
        assert set(o.keys()) == set(r.keys()) == set(info.keys())
        assert set(o.keys()).union({"__all__"}) == set(d.keys())
        tracks.append(d)
        if d[v_id_0]:
            assert info[v_id_0]["out_of_road"]
            assert info[v_id_0]["cost"] == out_of_road_cost
            assert r[v_id_0] == -out_of_road_penalty
            v_id_0 = "agent{}".format(count)
            count += 1
            done_count += 1
        if d[v_id_1]:
            assert info[v_id_1]["out_of_road"]
            assert info[v_id_1]["cost"] == out_of_road_cost
            assert r[v_id_1] == -out_of_road_penalty
            v_id_1 = "agent{}".format(count)
            count += 1
            done_count += 1
        if all(d.values()):
            raise ValueError()
        if i % 100 == 0:  # Horizon
            v_id_0 = "agent0"
            v_id_1 = "agent1"
            count = 2
            o = env.reset()
            assert set(o.keys()) == {"agent0", "agent1"}
            assert set(env.observations.keys()) == {"agent0", "agent1"}
            assert set(env.action_space.spaces.keys()) == {"agent0", "agent1"}
            assert set(env.config.target_vehicle_configs.keys()) == {"agent0", "agent1"}
            assert set(env.vehicles.keys()) == {"agent0", "agent1"}
    env.close()
    assert done_count > 0
    print("Finish {} dones.".format(done_count))


if __name__ == '__main__':
    setup_logger(True)
    test_respawn()
