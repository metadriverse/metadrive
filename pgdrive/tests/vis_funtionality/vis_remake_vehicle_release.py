from pgdrive.component.vehicle.base_vehicle import BaseVehicle
from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.utils import setup_logger

setup_logger(debug=True)


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__({
            "manual_control": True,
            "use_render": False,
        })

    def reset(self):
        if self.vehicles is not None:
            self.vehicle.destroy()
            self.vehicles["default_agent"] = BaseVehicle(env.engine)
            if self.main_camera is not None:
                self.main_camera.track(self.vehicle, env.engine)
        super(TestEnv, self).reset()


if __name__ == "__main__":
    env = TestEnv()

    env.reset()
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        # env.render("Test: {}".format(i))
        if d:
            env.reset()
    env.close()
