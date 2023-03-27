import sys
import os

import cv2
from metadrive import MetaDrive_PACKAGE_DIR
from metadrive.component.vehicle_module.mini_map import MiniMap
from metadrive.component.vehicle_module.rgb_camera import RGBCamera
from metadrive.component.vehicle_module.vehicle_panel import VehiclePanel
from metadrive.envs.metadrive_env import MetaDriveEnv


def capture_headless_image(cuda, image_source="main_camera"):
    env = MetaDriveEnv(
        dict(
            use_render=False,
            start_seed=666,
            image_on_cuda=cuda,
            traffic_density=0.1,
            image_observation=True,
            interface_panel=[MiniMap, RGBCamera, VehiclePanel],
            vehicle_config={
                "image_source": image_source,
                "rgb_camera": (512, 512),
                "depth_camera": (512, 512, False)
            },
        )
    )
    try:
        env.reset()
        for i in range(10):
            o, r, d, i = env.step([0, 1])
        assert isinstance(o, dict)
        # print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in o.items()})
        o = o["image"][..., -1] * 255 if not cuda else o["image"].get()[..., -1] * 255
        cv2.imwrite(
            os.path.join(
                MetaDrive_PACKAGE_DIR, "examples",
                "{}_from_observation{}.png".format(image_source, "_cuda" if cuda else "")
            ), o
        )
        cam = env.vehicle.get_camera(image_source)
        cam.save_image(
            env.vehicle,
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
            o, r, d, info = env.step([0, 1])
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
