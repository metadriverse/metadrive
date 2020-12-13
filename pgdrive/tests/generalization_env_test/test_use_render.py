import time

from pgdrive.envs.generalization_racing import GeneralizationRacing
from pgdrive.scene_creator.algorithm.BIG import BigGenerateMethod
from pgdrive.scene_creator.map import Map

if __name__ == "__main__":
    env = GeneralizationRacing(
        dict(
            use_render=True,
            map_config={
                Map.GENERATE_METHOD: BigGenerateMethod.BLOCK_NUM,
                Map.GENERATE_PARA: 7
            },
            traffic_density=0.5,
            manual_control=True,
            traffic_mode=0
        )
    )

    start = time.time()
    env.reset()
    env.render()
    print("Render cost time: ", time.time() - start)
    while True:
        o, r, d, info = env.step([0, 1])
        env.render()
        # if d:
        #     env.reset()
    env.close()
