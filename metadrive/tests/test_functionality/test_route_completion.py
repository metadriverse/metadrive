from metadrive.envs import MetaDriveEnv

from metadrive.policy.idm_policy import IDMPolicy
from metadrive.utils import setup_logger


def test_route_completion_easy():
    """
    Use an easy map to test whether the route completion is computed correctly.
    """
    # In easy map
    config = {}
    config["map"] = "SSS"
    config["traffic_density"] = 0
    try:
        env = MetaDriveEnv(config=config)
        o, i = env.reset()
        assert "route_completion" in i
        rc = env.vehicle.navigation.route_completion
        epr = 0
        for _ in range(1000):
            o, r, tm, tc, i = env.step([0, 1])
            epr += r
            env.render(mode="topdown")
            if tm or tc:
                epr = 0
                break
        assert "route_completion" in i
        print(i["route_completion"])
        assert i["route_completion"] > 0.95
    finally:
        if "env" in locals():
            env.close()


def test_route_completion_hard():
    """
    Use a hard map to test whether the route completion is computed correctly.
    """
    # In hard map
    config = {}
    config["map"] = "SCXTO"
    config["agent_policy"] = IDMPolicy
    config["traffic_density"] = 0
    try:
        env = MetaDriveEnv(config=config)
        o, i = env.reset()
        assert "route_completion" in i
        rc = env.vehicle.navigation.route_completion
        epr = 0
        for _ in range(1000):
            o, r, tm, tc, i = env.step([0, 0])
            epr += r
            env.render(mode="topdown")
            if tm or tc:
                epr = 0
                break
        assert "route_completion" in i
        print(i["route_completion"], i)
        assert i["route_completion"] > 0.8  # The vehicle will not reach destination due to randomness in IDM.
    finally:
        if "env" in locals():
            env.close()


if __name__ == '__main__':
    setup_logger(True)
    test_route_completion_hard()
