"""
We find running rendering (windows or RGB camera) with PG env with multiple scenarios has severe memory leak.
"""
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs.metadrive_env import MetaDriveEnv


def test_safe_env_memory_leak():
    # Initialize environment
    train_env_config = dict(
        # manual_control=False,  # Allow receiving control signal from external device
        # window_size=(200, 200),
        horizon=1500,
        # use_render=vis,
        image_observation=True,
        sensors=dict(rgb_camera=(RGBCamera, 256, 256)),
        num_scenarios=100,
    )

    env = MetaDriveEnv(train_env_config)
    try:
        total_cost = 0
        for ep in range(20):
            o, _ = env.reset()
            env.engine.force_fps.disable()
            for i in range(1, 1000):
                o, r, tm, tc, info = env.step([0, 1])
                total_cost += info["cost"]
                assert env.observation_space.contains(o)
                if tm or tc:
                    total_cost = 0
                    break
    finally:
        env.close()


if __name__ == '__main__':
    test_safe_env_memory_leak()
