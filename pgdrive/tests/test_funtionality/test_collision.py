from pgdrive.envs.pgdrive_env import PGDriveEnv


def test_collision_with_vehicle():
    env = PGDriveEnv({"traffic_density": 1.0})
    o = env.reset()
    pass_test = False
    try:
        for i in range(1, 100):
            o, r, d, info = env.step([0, 1])
            if env.vehicle.crash_vehicle:
                pass_test = True
                break
        assert pass_test, "Collision function is broken!"
    finally:
        env.close()


def test_collision_with_sidewalk():
    env = PGDriveEnv({"traffic_density": .0})
    o = env.reset()
    pass_test = False
    try:
        for i in range(1, 100):
            o, r, d, info = env.step([-0.5, 1])
            if env.vehicle.crash_sidewalk:
                pass_test = True
                break
        assert pass_test, "Collision function is broken!"
    finally:
        env.close()
