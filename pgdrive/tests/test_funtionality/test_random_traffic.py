from pgdrive.envs.pgdrive_env_v2 import PGDriveEnvV2

from pgdrive.utils import norm


def test_random_traffic():
    env = PGDriveEnvV2({
        "random_traffic": True,
        "traffic_mode": "respawn",
        # "fast": True, "use_render": True
    })
    try:
        last_pos = None
        for i in range(20):
            obs = env.reset()
            assert env.pgdrive_engine.traffic_manager.random_traffic
            assert env.pgdrive_engine.traffic_manager.random_seed is None
            new_pos = [v.position for v in env.pgdrive_engine.traffic_manager.vehicles]
            if last_pos is not None and len(new_pos) == len(last_pos):
                assert sum(
                    [norm(lastp[0] - newp[0], lastp[1] - newp[1]) >= 0.5 for lastp, newp in zip(last_pos, new_pos)]
                ), [(lastp, newp) for lastp, newp in zip(last_pos, new_pos)]
            last_pos = new_pos
    finally:
        env.close()


if __name__ == '__main__':
    test_random_traffic()
