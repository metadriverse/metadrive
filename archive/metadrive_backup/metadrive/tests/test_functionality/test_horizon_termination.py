from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
import pytest
import itertools

envs = [MetaDriveEnv, ScenarioEnv, MultiAgentMetaDrive]
horizon = [20, 200, None]
truncate_as_terminate = [True, False]

cfgs = itertools.product(envs, horizon, truncate_as_terminate)


@pytest.mark.parametrize("cfg", cfgs)
def test_horizon(cfg, use_render=False):
    env_cls = cfg[0]
    is_marl = env_cls == MultiAgentMetaDrive
    horizon = cfg[1]
    truncate_as_terminate = cfg[2]
    env = env_cls(
        {
            "use_render": use_render,
            "horizon": horizon,
            "num_scenarios": 1,
            "truncate_as_terminate": truncate_as_terminate
        }
    )
    o, _ = env.reset()
    test_pass = True if horizon is None else False
    try:
        for i in range(1, 500):
            action = {k: [0, -1] for k in env.agents.keys()} if is_marl else [0, -1]
            o, r, tms, tcs, infos = env.step(action)
            if not isinstance(tms, dict):
                tms = dict(__all__=tms)
                tcs = dict(__all__=tcs)
                infos = dict(__all__=infos)

            if tms["__all__"] or tcs["__all__"]:
                break

        for key in tcs.keys():
            if is_marl and key == "__all__":
                continue
            tc = tcs[key]
            info = infos[key]
            tm = tms[key]
            if tc:
                assert info["max_step"]
                assert info["episode_length"] == horizon
                test_pass = True
            if horizon:
                assert tc, cfg
                if truncate_as_terminate:
                    assert tm, cfg
                else:
                    # in single-agent, they are different
                    assert not tm, cfg
            else:
                assert not tc and not tm, cfg
        assert test_pass
    finally:
        env.close()


if __name__ == '__main__':
    test_horizon(iter(cfgs).__next__())
    # test_collision_with_vehicle(True)
