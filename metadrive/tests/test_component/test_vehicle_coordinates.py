import numpy as np

from metadrive.envs.metadrive_env import MetaDriveEnv


def test_coordinates_shift():
    try:
        env = MetaDriveEnv(
            {
                "num_scenarios": 100,
                "traffic_density": .0,
                "traffic_mode": "trigger",
                "start_seed": 22,
                # "manual_control": True,
                # "use_render": True,
                "decision_repeat": 5,
                "norm_pixel": True,
                "pstats": True,
                # "discrete_action": True,
                "map": "SSSSSS",
            }
        )
        env.reset()
        env.agent.set_velocity([1, 0], 10)
        # print(env.agent.speed)
        pos = [(x, y) for x in [-10, 0, 10] for y in [-20, 0, 20]] * 10
        p = pos.pop()
        for s in range(1, 100000):
            o, r, tm, tc, info = env.step([1, 0.3])
            if s % 10 == 0:
                if len(pos) == 0:
                    break
                p = pos.pop()
            p = np.asarray(p)
            heading, side = env.agent.convert_to_local_coordinates(p, env.agent.position)
            recover_pos = env.agent.convert_to_world_coordinates([heading, side], env.agent.position)
            if abs(recover_pos[0] - p[0]) + abs(recover_pos[1] - p[1]) > 0.1:
                raise ValueError("vehicle coordinates convert error!")
            if tm:
                break
    finally:
        env.close()


if __name__ == "__main__":
    test_coordinates_shift()
