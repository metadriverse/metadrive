import time

from pg_drive.envs.generalization_racing import GeneralizationRacing
from pg_drive.scene_creator.algorithm.BIG import BigGenerateMethod
from pg_drive.scene_creator.map import Map

if __name__ == "__main__":
    headless = False
    env = GeneralizationRacing(
        dict(
            use_render=False,
            map_config={
                Map.GENERATE_METHOD: BigGenerateMethod.BLOCK_NUM,
                Map.GENERATE_PARA: 7
            },
            traffic_density=0.5,
            manual_control=True,
            traffic_mode=0,
            use_image=True,
            pg_world_config=dict(headless_rgb=headless)
        )
    )

    start = time.time()
    env.reset()
    print("Render cost time: ", time.time() - start)
    for i in range(30):
        o, r, d, info = env.step([0, 1])
        from panda3d.core import PNMImage

        img = PNMImage()
        env.pg_world.win.getScreenshot(img)
        img.write("{}_{}.png".format("headless" if headless else "local", i))
    env.close()
