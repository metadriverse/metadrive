import sys
from metadrive.component.sensors.depth_camera import DepthCamera
import os

import cv2
from metadrive import MetaDrive_PACKAGE_DIR
from metadrive.component.sensors.mini_map import MiniMap
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.dashboard import DashBoard
from metadrive.envs.metadrive_env import MetaDriveEnv


def capture_headless_image(cuda, image_source="main_camera"):
    if image_source == "main_camera":
        sensors = {"main_camera": ()}
    elif image_source == "rgb_camera":
        sensors = {"rgb_camera": (RGBCamera, 256, 256)}
    elif image_source == "depth_camera":
        sensors = {"depth_camera": (DepthCamera, 256, 256)}
    else:
        sensors = {}
    env = MetaDriveEnv(
        dict(
            use_render=False,
            show_terrain="METADRIVE_TEST_EXAMPLE" not in os.environ,
            start_seed=666,
            image_on_cuda=cuda,
            traffic_density=0.1,
            image_observation=True,
            window_size=(600, 400),
            sensors=sensors,
            interface_panel=[],
            vehicle_config={
                "image_source": image_source,
            },
        )
    )
    try:
        env.reset()
        for i in range(10):
            o, r, tm, tc, i = env.step([0, 1])
        assert isinstance(o, dict)
        # print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in o.items()})
        o = o["image"][..., -1] * 255 if not cuda else o["image"].get()[..., -1] * 255
        cv2.imwrite(
            os.path.join(
                MetaDrive_PACKAGE_DIR, "examples",
                "{}_from_observation{}.png".format(image_source, "_cuda" if cuda else "")
            ), o
        )
        cam = env.engine.get_sensor(image_source)
        cam.save_image(
            env.agent,
            os.path.join(
                MetaDrive_PACKAGE_DIR, "examples", "{}_from_buffer{}.png".format(image_source, "_cuda" if cuda else "")
            )
        )
        # if image_source == "main_camera":
        #     ret = PNMImage()
        #     env.engine.win.getDisplayRegion(6).camera.node().getDisplayRegion(0).getScreenshot(ret)
        #     ret.write("test_1.png")
        #     new_ret = PNMImage()
        #     RGBCamera._singleton.buffer.getDisplayRegion(1).getScreenshot(new_ret)
        #     new_ret.write("test_2.png")
        print(
            "{} Test result: \nHeadless mode Offscreen render launched successfully! \n"
            "images named \'{}_from_observation.png\' and \'{}_from_buffer.png\' are saved to {}. "
            "Open it to check if offscreen mode works well".format(
                image_source, image_source, image_source, os.path.join(MetaDrive_PACKAGE_DIR, "examples")
            )
        )
    finally:
        env.close()


def verify_installation(cuda=False, camera="main"):
    env = MetaDriveEnv({"use_render": False, "image_observation": False})
    try:
        env.reset()
        for i in range(1, 100):
            o, r, tm, tc, info = env.step([0, 1])
    except:
        print("Error happens in Bullet physics world !")
        sys.exit()
    else:
        print("Bullet physics world is launched successfully!")
    finally:
        env.close()
        if camera == "main":
            capture_headless_image(cuda)
        elif camera == "rgb":
            capture_headless_image(cuda, image_source="rgb_camera")
        elif camera == "depth":
            capture_headless_image(cuda, image_source="depth_camera")
        else:
            raise ValueError("Can not find camera: {}, please select from [rgb, depth, main]".format(camera))


if __name__ == "__main__":
    verify_installation(camera="rgb")
