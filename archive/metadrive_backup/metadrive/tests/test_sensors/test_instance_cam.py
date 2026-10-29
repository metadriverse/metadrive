import pytest

from metadrive.component.sensors.instance_camera import InstanceCamera
from metadrive.envs.metadrive_env import MetaDriveEnv
import numpy as np
blackbox_test_configs = dict(
    # standard=dict(stack_size=3, width=256, height=128, norm_pixel=True),
    small=dict(stack_size=1, width=64, height=32, norm_pixel=True),
)


@pytest.mark.parametrize("config", list(blackbox_test_configs.values()), ids=list(blackbox_test_configs.keys()))
def test_instance_cam(config, render=False):
    """
    Test the output shape of Instance camera. This can NOT make sure the correctness of rendered image but only for
    checking the shape of image output and image retrieve pipeline
    Args:
        config: test parameter
        render: render with cv2

    Returns: None

    """
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.1,
            "map": "S",
            "show_terrain": False,
            "start_seed": 4,
            "stack_size": config["stack_size"],
            "vehicle_config": dict(image_source="camera"),
            "sensors": {
                "camera": (InstanceCamera, config["width"], config["height"])
            },
            "interface_panel": ["dashboard", "camera"],
            "image_observation": True,  # it is a switch telling metadrive to use rgb as observation
            "norm_pixel": config["norm_pixel"],  # clip rgb to range(0,1) instead of (0, 255)
        }
    )
    try:
        env.reset()
        base_free = len(env.engine.COLORS_FREE)
        base_occupied = len(env.engine.COLORS_OCCUPIED)
        assert base_free + base_occupied == env.engine.MAX_COLOR
        import cv2
        import time
        start = time.time()
        for i in range(1, 10):
            o, r, tm, tc, info = env.step([0, 1])
            assert env.observation_space.contains(o)
            # Reverse
            assert o["image"].shape == (
                config["height"], config["width"], InstanceCamera.num_channels, config["stack_size"]
            )
            image = o["image"][..., -1]
            image = image.reshape(-1, 3)
            unique_colors = np.unique(image, axis=0)
            assert len(unique_colors) > 0
            #Making sure every color observed correspond to an object
            for unique_color in unique_colors:
                if (unique_color != np.array((0, 0, 0))).all():  #Ignore the black background.
                    color = unique_color.tolist()
                    color = (
                        round(color[2], 5), round(color[1], 5), round(color[0], 5)
                    )  #In engine, we use 5-digit float for keys
                    if color not in env.engine.COLORS_OCCUPIED:
                        import matplotlib.pyplot as plt
                        plt.imshow(o["image"][..., -1])
                        plt.show()
                        print("Unique colors:", unique_colors)
                        print("Occupied colors:", env.engine.COLORS_OCCUPIED)
                    assert color in env.engine.COLORS_OCCUPIED
                    assert color not in env.engine.COLORS_FREE
                    assert color in env.engine.c_id.keys()
                    assert env.engine.id_c[env.engine.c_id[color]] == color  #Making sure the color-id is a bijection
                    assert len(env.engine.c_id.keys()) == len(env.engine.COLORS_OCCUPIED)
                    assert len(env.engine.id_c.keys()) == len(env.engine.COLORS_OCCUPIED)
                    assert len(env.engine.COLORS_FREE) + len(env.engine.COLORS_OCCUPIED) == env.engine.MAX_COLOR
            #Making sure every object in the engine(not necessarily observable) have corresponding color
            for id, object in env.engine.get_objects().items():
                assert id in env.engine.id_c.keys()
            if render:
                cv2.imshow('img', o["image"][..., -1])
                cv2.waitKey(1)
        print("FPS:", 10 / (time.time() - start))
    finally:
        env.close()


if __name__ == '__main__':
    test_instance_cam(config=blackbox_test_configs["small"], render=True)
    # my_dict = {(0, 0, 0): "Hello, World!"}
