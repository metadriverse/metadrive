from pg_drive.envs.generalization_racing import GeneralizationRacing
from pg_drive.scene_creator.algorithm.BIG import BigGenerateMethod
from pg_drive.scene_manager.traffic_manager import TrafficMode
from pg_drive.utils import setup_logger

setup_logger(debug=True)

if __name__ == "__main__":
    env = GeneralizationRacing(
        dict(
            use_render=True,
            manual_control=True,
            traffic_density=0.2,
            traffic_mode=TrafficMode.Reborn,
            environment_num=100,
            map_config={
                "type": BigGenerateMethod.BLOCK_NUM,
                "config": 7,
            },
        )
    )
    print("Press h to see help message!")
    env.reset()
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 0])
        env.render()
    env.close()
