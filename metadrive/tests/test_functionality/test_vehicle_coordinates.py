from metadrive.envs.metadrive_env import MetaDriveEnv


def test_coordinates_shift():
    try:
        env = MetaDriveEnv(
            {
                "environment_num": 100,
                "traffic_density": .0,
                "traffic_mode": "trigger",
                "start_seed": 22,
                # "manual_control": True,
                # "use_render": True,
                "decision_repeat": 5,
                "rgb_clip": True,
                "pstats": True,
                # "discrete_action": True,
                "map": "SSSSSS",
            }
        )
        env.reset()
        env.vehicle.set_velocity([1, 0], 10)
        print(env.vehicle.speed)
        pos = [(x, y) for x in [-10, 0, 10] for y in [-20, 0, 20]] * 10
        p = pos.pop()
        for s in range(1, 100000):
            o, r, d, info = env.step([1, 0.3])
            if s % 10 == 0:
                if len(pos) == 0:
                    break
                p = pos.pop()
            heading, side = env.vehicle.convert_to_vehicle_coordinates(p)
            recover_pos = env.vehicle.convert_to_world_coordinates(heading, side)
            if abs(recover_pos[0] - p[0]) + abs(recover_pos[1] - p[1]) > 0.1:
                raise ValueError("vehicle coordinates convert error!")
    finally:
        env.close()


if __name__ == "__main__":
    test_coordinates_shift()
