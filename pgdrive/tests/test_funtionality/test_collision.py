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


def test_line_contact():
    env = PGDriveEnv({"traffic_density": .0})
    o = env.reset()
    on_broken_line = False
    on_continuous_line = False
    try:
        for i in range(1, 100):
            o, r, d, info = env.step([-0.5, 1])
            on_broken_line = on_broken_line or env.vehicle.on_broken_line
            on_continuous_line = on_continuous_line or env.vehicle.on_white_continuous_line
        assert on_broken_line and on_continuous_line, "Collision function is broken!"
    finally:
        env.close()
