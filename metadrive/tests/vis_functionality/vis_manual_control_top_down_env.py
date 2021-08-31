from metadrive.envs.top_down_env import TopDownMetaDrive

if __name__ == '__main__':
    env = TopDownMetaDrive(
        dict(
            # environment_num=100,
            use_render=True,
            # manual_control=True,

            # Extremely easy environment
            environment_num=100,
            traffic_density=0.0,
            # map="S",
            start_seed=5000,
            # frame_stack=tune.grid_search([1, 3, 5, 7]),
            # frame_stack=tune.grid_search([1, 3, 5]),
        )
    )
    env.reset()
    try:
        ep_reward = 0
        while True:
            o, r, d, i = env.step([0.01, 1])
            print("Obs shape {}, reward {:.8f}, done {}, info {}".format(o.shape, r, d, i))
            env.render()

            ep_reward += r
            if d:
                print("Episode reward: ", ep_reward)
                ep_reward = 0
                env.reset()

    finally:
        env.close()
