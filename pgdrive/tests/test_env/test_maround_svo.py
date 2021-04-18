from pgdrive.envs.marl_envs.maround_svo import MARoundSVO


def test_maround_svo():
    configs = [
        {
            "num_agents": 8,
            "num_neighbours": 8,
            "svo_mode": "angle",
            "horizon": 200
        }, {
            "num_agents": 2,
            "num_neighbours": 8,
            "svo_mode": "linear",
            "horizon": 200
        }, {
            "num_agents": 8,
            "num_neighbours": 0,
            "svo_mode": "angle",
            "horizon": 200
        }, {
            "num_agents": 1,
            "num_neighbours": 0,
            "svo_mode": "angle",
            "horizon": 200
        }
    ]
    for c in configs:
        env = MARoundSVO(c)
        try:
            o = env.reset()
            assert env.observation_space.contains(o)
            assert all([0 <= oo[-1] <= 1.0 for oo in o.values()])
            total_r = 0
            ep_s = 0
            for i in range(1, 100000):
                o, r, d, info = env.step({k: [0.0, 1.0] for k in env.vehicles.keys()})
                assert env.observation_space.contains(o)
                assert all([0 <= oo[-1] <= 1.0 for oo in o.values()])
                for r_ in r.values():
                    total_r += r_
                ep_s += 1
                if d["__all__"]:
                    break
        finally:
            env.close()
            print('Finish config: ', c)


if __name__ == '__main__':
    test_maround_svo()
