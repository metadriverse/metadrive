from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.scene_creator.map import Map, MapGenerateMethod
from pgdrive.utils import setup_logger
from pgdrive.scene_creator.ego_vehicle.vehicle_module.PID_controller import PIDController
from pgdrive.scene_creator.ego_vehicle.vehicle_module.routing_localization import RoutingLocalizationModule

setup_logger(True)


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 1,
                "traffic_density": 0.0,
                "use_render": True,
                "start_seed": 5,
                "map_config": {
                    Map.GENERATE_METHOD: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                    Map.GENERATE_PARA: "rTCROSXR",
                    Map.LANE_WIDTH: 3.5,
                    Map.LANE_NUM: 3,
                }
            }
        )


if __name__ == "__main__":

    class Target:
        def __init__(self, target_lateral, target_speed):
            self.lateral = target_lateral
            self.speed = target_speed

        def go_right(self):
            self.lateral += 0.25

        def go_left(self):
            self.lateral -= 0.25

        def faster(self):
            self.speed += 10

        def slower(self):
            self.speed -= 10

    env = TestEnv()
    target = Target(0.375, 30)
    o = env.reset()
    env.pg_world.accept('d', target.go_right)
    env.pg_world.accept('a', target.go_left)
    env.pg_world.accept('w', target.faster)
    env.pg_world.accept('s', target.slower)

    steering_controller = PIDController(1.6, 0.0008, 27.3)
    steering_error = o[0] - target.lateral
    steering = steering_controller.get_result(steering_error)
    acc_controller = PIDController(0.1, 0.001, 0.3)
    acc_error = env.vehicle.speed - target.speed
    acc = acc_controller.get_result(acc_error)
    for i in range(1, 100000):
        o, r, d, info = env.step([-steering, acc])
        # calculate new action

        steering_error = o[0] - target.lateral
        steering = steering_controller.get_result(steering_error)
        t_speed = target.speed if abs(o[12] - 0.5) < 0.01 else target.speed - 10
        acc_error = env.vehicle.speed - t_speed
        acc = acc_controller.get_result(acc_error)

        env.render()
        if d:
            print("Reset")
            o = env.reset()
            steering_controller.reset()
            steering_error = o[0] - target.lateral
            steering = steering_controller.get_result(steering_error, o[11])
            acc_controller.reset()
            acc_error = env.vehicle.speed - target.speed
            acc = acc_controller.get_result(acc_error)
    env.close()
