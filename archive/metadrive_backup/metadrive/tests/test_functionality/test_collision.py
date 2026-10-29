from metadrive.envs.metadrive_env import MetaDriveEnv


def test_collision_with_vehicle(use_render=False):
    if not use_render:
        env = MetaDriveEnv({"traffic_density": 1.0, "map": "SSS"})
    else:
        env = MetaDriveEnv({"traffic_density": 1.0, "map": "SSS", "use_render": True})
    o, _ = env.reset()
    pass_test = False
    try:
        for i in range(1, 500):
            o, r, tm, tc, info = env.step([0, 1])
            if env.agent.crash_vehicle:
                pass_test = True
                break
        assert pass_test, "Collision function is broken!"
    finally:
        env.close()


def test_collision_with_sidewalk():
    env = MetaDriveEnv({"traffic_density": .0})
    o, _ = env.reset()
    pass_test = False
    try:
        for i in range(1, 100):
            o, r, tm, tc, info = env.step([-0.5, 1])
            if env.agent.crash_sidewalk:
                pass_test = True
                break
        assert pass_test, "Collision function is broken!"
    finally:
        env.close()


def test_line_contact():
    env = MetaDriveEnv({"traffic_density": .0})
    o, _ = env.reset()
    on_broken_line = False
    on_continuous_line = False
    try:
        for i in range(1, 100):
            o, r, tm, tc, info = env.step([-0.5, 1])
            on_broken_line = on_broken_line or env.agent.on_broken_line
            on_continuous_line = on_continuous_line or env.agent.on_white_continuous_line
        assert on_broken_line and on_continuous_line, "Collision function is broken!"
    finally:
        env.close()


def test_collision_info(use_render=False):
    env = MetaDriveEnv({"traffic_density": 1.0, "map": "SSS", "use_render": use_render, "crash_vehicle_done": False})
    o, _ = env.reset()
    pass_test = False
    try:
        for i in range(1, 500):
            o, r, tm, tc, info = env.step([0, 1])
            if info["crash_vehicle"]:
                pass_test = True
                break
        assert pass_test, "Collision info is not in info!"
    finally:
        env.close()


if __name__ == '__main__':
    test_collision_with_sidewalk()
    # test_collision_with_vehicle(True)
