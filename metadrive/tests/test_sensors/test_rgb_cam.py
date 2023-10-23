import pytest

from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs.metadrive_env import MetaDriveEnv

blackbox_test_configs = dict(
    standard=dict(stack_size=3, width=256, height=128, rgb_clip=True),
    large=dict(stack_size=5, width=800, height=600, rgb_clip=True),
    no_clip=dict(stack_size=3, width=800, height=600, rgb_clip=False),
)


@pytest.mark.parametrize("config", list(blackbox_test_configs.values()), ids=list(blackbox_test_configs.keys()))
def test_rgb_cam(config, render=False):
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.1,
            "map": "S",
            "start_seed": 4,
            "stack_size": config["stack_size"],
            "vehicle_config": dict(image_source="rgb_camera"),
            "sensors": {
                "rgb_camera": (RGBCamera, config["width"], config["height"])
            },
            "interface_panel": ["dashboard", "rgb_camera"],
            "image_observation": True,  # it is a switch telling metadrive to use rgb as observation
            "rgb_clip": config["rgb_clip"],  # clip rgb to range(0,1) instead of (0, 255)
        }
    )
    env.reset()
    try:
        import cv2
        for i in range(1, 10):
            o, r, tm, tc, info = env.step([0, 1])
            assert env.observation_space.contains(o)
            assert o["image"].shape == (config["height"], config["width"], 3, config["stack_size"])
            if render:
                cv2.imshow('img', o["image"][..., -1])
                cv2.waitKey(1)
    finally:
        env.close()


if __name__ == '__main__':
    test_rgb_cam(config=blackbox_test_configs["standard"], render=True)
