from pgdrive.constants import TerminationState
from pgdrive.envs.pgdrive_env import PGDriveEnv

if __name__ == "__main__":
    env = PGDriveEnv(
        {
            "environment_num": 1,
            "traffic_density": 0.1,
            "start_seed": 5,
            # "controller": "joystick",
            "manual_control": True,
            "use_render": True,
            "use_saver": True,
            "save_level": 0.3,
            "map": 3
        }
    )

    o = env.reset()
    env.engine.force_fps.toggle()
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        text = {"AI Intervene": info["takeover_start"]}
        env.render(text=text)
        if info[TerminationState.SUCCESS]:
            env.reset()
    env.close()
