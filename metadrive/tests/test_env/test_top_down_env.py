import numpy as np

from metadrive.envs.top_down_env import TopDownSingleFrameMetaDriveEnv, TopDownMetaDrive, TopDownMetaDriveEnvV2


def test_top_down_rendering():
    for env in [
            TopDownSingleFrameMetaDriveEnv(dict(environment_num=5, map="C", traffic_density=1.0)),
            TopDownMetaDrive(dict(environment_num=5, map="C", traffic_density=1.0)),
            TopDownMetaDrive(dict(environment_num=5, map="C", frame_stack=1, post_stack=2)),
            TopDownMetaDriveEnvV2(dict(environment_num=5, map="C", frame_stack=1, post_stack=2)),
    ]:
        try:
            for _ in range(5):
                o = env.reset()
                assert np.mean(o) > 0.0
                for _ in range(10):
                    o, *_ = env.step([0, 1])
                    assert np.mean(o) > 0.0
                for _ in range(10):
                    o, *_ = env.step([-0.05, 1])
                    assert np.mean(o) > 0.0
        finally:
            env.close()


if __name__ == "__main__":
    test_top_down_rendering()
