from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils import setup_logger


def test_traffic_respawn(vis=False):
    setup_logger(vis)

    env = MetaDriveEnv({"num_scenarios": 1, "manual_control": vis, "use_render": vis, "traffic_mode": "respawn"})
    env.reset()
    try:
        for i in range(1, 3000):
            env.step([0, 0])
            current_v = set(env.engine.traffic_manager.vehicles)
            for v in list(env.engine.traffic_manager.traffic_vehicles) + [env.agent]:
                if v is env.agent:
                    current_v.discard(v)
                else:
                    current_v.discard(v)
            assert len(current_v) == 0, "vehicles didn't release"
            assert len(env.engine.traffic_manager.vehicles) - len(env.engine.traffic_manager.traffic_vehicles) == 1, \
                "vehicles didn't release"
    finally:
        env.close()


if __name__ == '__main__':
    test_traffic_respawn(vis=False)
