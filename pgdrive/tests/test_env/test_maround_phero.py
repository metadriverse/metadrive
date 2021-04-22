from pgdrive.envs.marl_envs.maround_phero import MARoundPhero


def test_maround_phero():
    configs = [
        {
            "num_agents": 8,
            "num_neighbours": 9,
            "num_channels": 1,
            "horizon": 200
        },
        {
            "num_agents": 8,
            "num_neighbours": 1,
            "num_channels": 1,
            "horizon": 200
        },
        {
            "num_agents": 8,
            "num_neighbours": 9,
            "num_channels": 3,
            "horizon": 200
        },
        {
            "num_agents": 8,
            "num_neighbours": 1,
            "num_channels": 3,
            "horizon": 200
        },
        {
            "num_agents": 8,
            "num_neighbours": 9,
            "num_channels": 3,
            "attenuation_rate": 0.0,
            "diffusion_rate": 0.0,
            "horizon": 200
        },
    ]
    for c in configs:
        env = MARoundPhero(c)
        try:
            o = env.reset()
            assert env.observation_space.contains(o)
            assert all([0 <= oo[-1] <= 1.0 for oo in o.values()])
            total_r = 0
            ep_s = 0
            for i in range(1, 100000):
                if i % 2 == 0:
                    phe = [0.0] * c["num_channels"]
                    a = {k: [0.0, 1.0] + phe for k in env.vehicles.keys()}
                else:
                    a = env.action_space.sample()
                o, r, d, info = env.step(a)
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
    test_maround_phero()
