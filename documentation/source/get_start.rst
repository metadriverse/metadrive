.. _getting_start:

#############################
Getting Start with MetaDrive
#############################

We provide a pre-trained RL agent to show the power of MetaDrive.
Please run the following script to watch its performance::

    python -m metadrive.examples.enjoy_expert

You can also manually control a vehicle with keyboard, please run::

     python -m metadrive.examples.enjoy_manual

To enjoy the process of generate map through our Block Incremental Generation (BIG) algorithm, you can also run::

    python -m metadrive.examples.render_big

*Note that the above three scripts can not be run in headless machine.*

You can verify the efficiency of MetaDrive via running::

    python -m metadrive.examples.profile_metadrive

You can also draw multiple maps in the top-down view via running::

    python -m metadrive.examples.draw_maps

Environment Usage
#########################

The usage of MetaDrive is as same as other **gym** environments::

    import metadrive  # Import this package to register the environment!
    import gym

    env = gym.make("MetaDrive-v0", config=dict(use_render=True))
    env.reset()
    for i in range(1000):
        obs, reward, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            env.reset()
    env.close()

Any Reinforcement Learning algorithms and Imitation Learning algorithms are compatible with MetaDrive.

Pre-defined Environments
#############################
Besides, we provide several environments for different purposes.
The following table presents some predefined environment names. Please feel free to open an issue if you want to request some new environments.

+-------------------------+-------------------+----------------+---------------------------------------------------------+
| Gym Environment Name    | Random Seed Range | Number of Maps | Comments                                                |
+=========================+===================+================+=========================================================+
| `MetaDrive-test-v0`       | [0, 200)          | 200            | Test set, not change for all experiments.               |
+-------------------------+-------------------+----------------+---------------------------------------------------------+
| `MetaDrive-validation-v0` | [200, 1000)       | 800            | Validation set.                                         |
+-------------------------+-------------------+----------------+---------------------------------------------------------+
| `MetaDrive-v0`            | [1000, 1100)      | 100            | Default training setting, for quick start.              |
+-------------------------+-------------------+----------------+---------------------------------------------------------+
| `MetaDrive-10envs-v0`     | [1000, 1100)      | 10             | Training environment with 10 maps.                      |
+-------------------------+-------------------+----------------+---------------------------------------------------------+
| `MetaDrive-1000envs-v0`   | [1000, 1100)      | 1000           | Training environment with 1000 maps.                    |
+-------------------------+-------------------+----------------+---------------------------------------------------------+
| `MetaDrive-training0-v0`  | [3000, 4000)      | 1000           | First set of 1000 environments.                         |
+-------------------------+-------------------+----------------+---------------------------------------------------------+
| `MetaDrive-training1-v0`  | [5000, 6000)      | 1000           | Second set of 1000 environments.                        |
+-------------------------+-------------------+----------------+---------------------------------------------------------+
| `MetaDrive-training2-v0`  | [7000, 8000)      | 1000           | Thirds set of 1000 environments.                        |
+-------------------------+-------------------+----------------+---------------------------------------------------------+
| ...                     |                   |                | *More map set can be added in response to the requests* |
+-------------------------+-------------------+----------------+---------------------------------------------------------+


