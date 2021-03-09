from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.scene_creator.map import Map, MapGenerateMethod
from pgdrive.utils import setup_logger

setup_logger(True)


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__({"use_render": True, "manual_control": True, "environment_num": 100})


if __name__ == "__main__":
    env = TestEnv()

    o = env.reset()
    print("vehicle num", len(env.scene_manager.traffic_mgr.vehicles))
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        env.render(
            text={
                "vehicle_num": len(env.scene_manager.traffic_mgr.traffic_vehicles),
                "dist_to_left:": env.vehicle.dist_to_left,
                "dist_to_right:": env.vehicle.dist_to_right
            }
        )
        # if d:
        #     print("Reset")
        #     env.reset()
    env.close()
