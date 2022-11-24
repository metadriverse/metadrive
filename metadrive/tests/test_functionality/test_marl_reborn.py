from metadrive.constants import TerminationState
from metadrive.envs.marl_envs.marl_inout_roundabout import MultiAgentRoundaboutEnv
from metadrive.utils import setup_logger


def test_respawn():
    out_of_road_cost = 5555
    out_of_road_penalty = 2222
    env = MultiAgentRoundaboutEnv(
        {
            "num_agents": 2,
            "out_of_road_cost": out_of_road_cost,
            "out_of_road_penalty": out_of_road_penalty,
            "delay_done": 0,  # Since we are testing respawn!
            # "use_render": True
            "crash_done": False,
        }
    )
    try:
        assert set(env.observations.keys()) == {"agent0", "agent1"}
        assert set(env.action_space.spaces.keys()) == {"agent0", "agent1"}
        assert set(env.config["target_vehicle_configs"].keys()) == {"agent0", "agent1"}
        assert set(env.vehicles.keys()) == set()  # Not initialized yet!

        o = env.reset()

        assert set(o.keys()) == {"agent0", "agent1"}
        assert set(env.observations.keys()) == {"agent0", "agent1"}
        assert set(env.action_space.spaces.keys()) == {"agent0", "agent1"}
        assert set(env.config["target_vehicle_configs"].keys()) == {"agent0", "agent1"}
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
                assert info[v_id_0][TerminationState.OUT_OF_ROAD]
                assert info[v_id_0]["cost"] == out_of_road_cost
                assert r[v_id_0] == -out_of_road_penalty
                v_id_0 = "agent{}".format(count)
                count += 1
                done_count += 1
            if d[v_id_1]:
                assert info[v_id_1][TerminationState.OUT_OF_ROAD]
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
                assert set(env.config["target_vehicle_configs"].keys()) == {"agent0", "agent1"}
                assert set(env.vehicles.keys()) == {"agent0", "agent1"}
    finally:
        env.close()
    assert done_count > 0
    print("Finish {} dones.".format(done_count))


def test_delay_done(render=False):
    # Put agent 0 in the left, agent 1 in the right, and let agent 0 dead at first.
    # We wish to see agent 1 hits the dead body of agent 0.
    env = MultiAgentRoundaboutEnv(
        {
            # "use_render": True,
            #
            "target_vehicle_configs": {
                "agent0": {
                    "spawn_longitude": 12,
                    "spawn_lateral": 0,
                    "spawn_lane_index": (">", ">>", 0),
                },
                "agent1": {
                    "spawn_longitude": 10,  # locate a little forward
                    "spawn_lateral": 0,
                    "spawn_lane_index": (">", ">>", 1),
                }
            },
            "num_agents": 2,
            "traffic_density": 0,
            "delay_done": 100,
            "horizon": 100,
            "use_render": render
        }
    )
    try:
        agent0_done = False
        agent1_already_hit = False
        o = env.reset()
        for i in range(1, 300):
            actions = {"agent0": [1, 1], "agent1": [1, 1]}
            if "agent0" not in env.vehicles:
                actions.pop("agent0")
            if "agent1" not in env.vehicles:
                actions.pop("agent1")
            o, r, d, info = env.step(actions)
            if agent0_done:
                assert "agent0" not in o
                assert "agent0" not in info
                assert "agent0" not in d
            if d.get("agent0"):
                agent0_done = True
            if agent0_done:
                if info["agent1"][TerminationState.CRASH_VEHICLE]:
                    agent1_already_hit = True
                    print("Hit!")
            if d["__all__"]:
                assert agent1_already_hit
                agent0_done = False
                agent1_already_hit = False
                env.reset()
    finally:
        env.close()

    env = MultiAgentRoundaboutEnv({"num_agents": 5, "delay_done": 10, "horizon": 100})
    try:
        env.reset()
        dead = set()
        for _ in range(300):
            o, r, d, i = env.step({k: [1, 1] for k in env.vehicles.keys()})
            for dead_name in dead:
                assert dead_name not in o
            print("{} there!".format(env.vehicles.keys()))
            print("{} dead!".format([kkk for kkk, ddd in d.items() if ddd]))
            for kkk, ddd in d.items():
                if ddd and kkk != "__all__":
                    dead.add(kkk)
            if d["__all__"]:
                env.reset()
                dead.clear()
    finally:
        env.close()


if __name__ == '__main__':
    # setup_logger(True)
    # test_respawn()
    test_delay_done(True)
