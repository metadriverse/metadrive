from pg_drive.envs.generalization_racing import GeneralizationRacing
from pg_drive.scene_creator.map import Map, MapGenerateMethod
from pg_drive.utils import setup_logger
from pg_drive.scene_manager.traffic_manager import TrafficMode

setup_logger(debug=True)


class TestEnv(GeneralizationRacing):
    def __init__(self):
        super(TestEnv, self).__init__({
            "use_render": False,
            "use_image": False,
        })


if __name__ == "__main__":
    env = TestEnv()

    env.reset()
    for i in range(1, 100):
        o, r, d, info = env.step([0, 1])
    env.close()
    print("Physics world successfully run!")
