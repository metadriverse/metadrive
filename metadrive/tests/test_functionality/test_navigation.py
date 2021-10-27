from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.component.vehicle_module.PID_controller import PIDController
from metadrive.envs.metadrive_env import MetaDriveEnv


class Target:
    def __init__(self, target_lateral, target_speed):
        self.lateral = target_lateral
        self.speed = target_speed

    def go_right(self):
        self.lateral += 0.25 if self.lateral < 0.625 else 0

    def go_left(self):
        self.lateral -= 0.25 if self.lateral > 0.125 else 0

    def faster(self):
        self.speed += 10

    def slower(self):
        self.speed -= 10


def test_navigation(vis=False):
    env = MetaDriveEnv(
        {
            "environment_num": 10,
            "traffic_density": 0.0,
            "use_render": vis,
            "start_seed": 5,
            "map_config": {
                BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
                BaseMap.GENERATE_CONFIG: 7,
                BaseMap.LANE_WIDTH: 3.5,
                BaseMap.LANE_NUM: 3,
            }
        }
    )
    target = Target(0.375, 30)
    o = env.reset()
    if vis:
        env.engine.accept('d', target.go_right)
        env.engine.accept('a', target.go_left)
        env.engine.accept('w', target.faster)
        env.engine.accept('s', target.slower)

    steering_controller = PIDController(1.6, 0.0008, 27.3)
    acc_controller = PIDController(0.1, 0.001, 0.3)

    steering_error = o[0] - target.lateral
    steering = steering_controller.get_result(steering_error)

    acc_error = env.vehicles[env.DEFAULT_AGENT].speed - target.speed
    acc = acc_controller.get_result(acc_error)
    for i in range(1, 1000000 if vis else 2000):
        o, r, d, info = env.step([-steering, acc])
        # calculate new action

        steering_error = o[0] - target.lateral
        steering = steering_controller.get_result(steering_error)

        t_speed = target.speed if abs(o[12] - 0.5) < 0.01 else target.speed - 10
        acc_error = env.vehicles[env.DEFAULT_AGENT].speed - t_speed
        acc = acc_controller.get_result(acc_error)
        if vis:
            if i < 700:
                env.render(
                    text={
                        "W": "Target speed +",
                        "S": "Target speed -",
                        "A": "Change to left lane",
                        "D": "Change to right lane"
                    }
                )
            if i == 500:
                env.engine.on_screen_message.data.clear()
            else:
                env.render()
        if d:
            print("Reset")
            o = env.reset()

            steering_controller.reset()
            steering_error = o[0] - target.lateral
            steering = steering_controller.get_result(steering_error, o[11])

            acc_controller.reset()
            acc_error = env.vehicles[env.DEFAULT_AGENT].speed - target.speed
            acc = acc_controller.get_result(acc_error)
    env.close()


if __name__ == "__main__":
    test_navigation(vis=True)
