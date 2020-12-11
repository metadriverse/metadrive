import time

from panda3d.core import PNMImage

from pg_drive.envs.generalization_racing import GeneralizationRacing
from pg_drive.scene_creator.algorithm.BIG import BigGenerateMethod
from pg_drive.scene_creator.map import Map


def capture_image(headless):
    env = GeneralizationRacing(
        dict(
            use_render=False,
            map_config={
                Map.GENERATE_METHOD: BigGenerateMethod.BLOCK_NUM,
                Map.GENERATE_PARA: 7
            },
            traffic_density=0.5,
            use_image=True,
            pg_world_config=dict(headless_image=headless)
        )
    )
    start = time.time()
    env.reset()
    print("Render cost time: ", time.time() - start)
    for i in range(3):
        o, r, d, info = env.step([0, 1])
        img = PNMImage()
        env.pg_world.win.getScreenshot(img)
        img.write("{}_{}.png".format("headless" if headless else "local", i))
    env.close()


if __name__ == "__main__":
    # This file should generate images in headless machine in headless=True mode or others in headless=False mode.
    headless = False
    capture_image(headless)
    print("Offscreen render launched successfully! Images are saved, Please check them.")
