from pgdrive.envs.pgdrive_env import PGDriveEnv
import pygame


def test_top_down_rendering():
    env = PGDriveEnv(dict(environment_num=20, start_seed=0, map=10, use_topdown=True, use_render=False))
    try:
        env.reset()
        for i in range(5):
            env.step(env.action_space.sample())
            env.render(mode="human")
            env.render(mode="rgb_array")
    finally:
        pygame.image.save(env.pg_world.highway_render.frame_surface, "save_offscreen.jpg")
        env.close()


if __name__ == "__main__":
    test_top_down_rendering()
