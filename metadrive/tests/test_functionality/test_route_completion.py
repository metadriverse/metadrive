from metadrive.envs import MetaDriveEnv
from metadrive.utils import setup_logger


def test_route_completion():
    # Crash vehicle
    config = {}
    config["map"] = "SSS"
    config["traffic_density"] = 0
    try:
        env = MetaDriveEnv(config=config)
        o, i = env.reset()
        rc = env.vehicle.navigation.route_completion
        epr = 0
        for _ in range(1000):
            o, r, tm, tc, i = env.step([0, 1])
            epr += r
            env.render(mode="topdown")
            # print("R: {}, Accu R: {}".format(r, epr))
            if tm or tc:
                epr = 0
                break
        # assert i[TerminationState.CRASH]
        # assert i[TerminationState.CRASH_VEHICLE]
        print(1111)
    finally:
        if "env" in locals():
            env.close()


if __name__ == '__main__':
    setup_logger(True)
    test_route_completion()
