import os
import sys

import cv2
from PIL import Image
from panda3d.core import PNMImage

from metadrive.envs.metadrive_env import MetaDriveEnv


def capture_headless_image(headless):
    env = MetaDriveEnv(
        dict(
            use_render=False,
            start_seed=666,
            traffic_density=0.1,
            offscreen_render=True,
            vehicle_config={"image_source": "main_camera"},
            # headless_machine_render=headless
        )
    )
    env.reset()
    for i in range(10):
        o, r, d, i = env.step([0, 1])
    assert isinstance(o, dict)
    print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in o.items()})
    o = o["image"][..., -1]*255
    cv2.imwrite("image_from_observation.png", o)
    img = PNMImage()
    env.engine.win.getScreenshot(img)
    img.write("main_camera.png")
    env.close()
    if not headless:
        im = Image.open("vis_installation.png")
        im.show()
        os.remove("vis_installation.png")
        print("Offscreen render launched successfully! \n ")
    else:
        print(
            "Headless mode Offscreen render launched successfully! \n"
            "images named \'main_camera.png\' and \'image_from_observation.png\' are saved. "
            "Open it to check if offscreen mode works well"
        )
    env.close()


def verify_installation(headless=True):
    env = MetaDriveEnv({"use_render": False, "offscreen_render": False})
    try:
        env.reset()
        for i in range(1, 100):
            o, r, d, info = env.step([0, 1])
        env.close()
        del env
    except:
        print("Error happens in Bullet physics world !")
        sys.exit()
    else:
        print("Bullet physics world is launched successfully!")
    capture_headless_image(headless)


if __name__ == "__main__":
    verify_installation(False)
