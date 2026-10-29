import numpy as np

from metadrive.envs import VaryingDynamicsEnv


def test_varying_dynamics_env():
    env = VaryingDynamicsEnv({"num_scenarios": 10})
    try:
        dys = []
        for seed in range(10):
            env.reset(seed=seed)
            for _ in range(10):
                env.step(env.action_space.sample())
            dy = env.agent.get_dynamics_parameters()
            print("Dynamics: ", dy)
            dy = np.array(list(dy.values()))
            if len(dys) > 0:
                for dyy in dys:
                    assert not np.all(dy == dyy)
            dys.append(dy)
    finally:
        env.close()


if __name__ == "__main__":
    test_varying_dynamics_env()
