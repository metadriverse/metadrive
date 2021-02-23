from pgdrive.envs.generation_envs.side_pass_env import SidePassEnv

# setup_logger(True)


def test_collision(render):
    env = SidePassEnv({"manual_control": render, "use_render": render, "debug": True})
    o = env.reset()

    for i in range(1, 100000 if render else 2000):
        o, r, d, info = env.step([0, 1])
        env.render()
        # if d:
        #     break
    env.close()


if __name__ == "__main__":
    test_collision(True)
