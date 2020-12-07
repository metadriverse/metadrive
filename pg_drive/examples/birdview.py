from pg_drive.envs.generalization_racing import GeneralizationRacing
from pg_drive.scene_creator.algorithm.BIG import BigGenerateMethod
from pg_drive.utils import setup_logger

setup_logger(debug=True)

if __name__ == "__main__":
    env = GeneralizationRacing(
        dict(
            map_config={
                "type": BigGenerateMethod.BLOCK_NUM,
                "config": 7,
            },
            camera_height=1000.0,
            use_render=True,
            environment_num=100,
            traffic_density=0.0,
        )
    )
    env.reset()
    for i in range(1, 100000):
        o, r, d, info = env.step([0.1, 0])
        env.render()
        if i % 10 == 0:
            env.reset()
    env.close()
