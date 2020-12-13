.. _getting_start:

#############################
Getting Start with PGDrive
#############################

We provide a pre-trained RL-Agent to show the power of PGDrive.
Run::

    python -m pgdrive.examples.enjoy_journey

to watch its show!

Environment Usage
#########################

The usage of PGDrive is as same as other **gym** environments::

    import pgdrive  # Import this package to register the environment!
    import gym

    env = gym.make("PGDrive-v0", config=dict(use_render=True))
    env.reset()
    for i in range(1000):
        obs, reward, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            env.reset()
    env.close()

Any Reinforcement Algorithms and Imitation Learning Algorithms are compatible with PGDrive.

Pre-defined Environments
#############################
Besides, we provide several environments for different purposes.
The following table presents some predefined environment names. Please feel free to open an issue if you want to request some new environments.

+-------------------------+-------------------+----------------+---------------------------------------------------------+
| Gym Environment Name    | Random Seed Range | Number of Maps | Comments                                                |
+=========================+===================+================+=========================================================+
| `PGDrive-test-v0`       | [0, 200)          | 200            | Test set, not change for all experiments.               |
+-------------------------+-------------------+----------------+---------------------------------------------------------+
| `PGDrive-validation-v0` | [200, 1000)       | 800            | Validation set.                                         |
+-------------------------+-------------------+----------------+---------------------------------------------------------+
| `PGDrive-v0`            | [1000, 1100)      | 100            | Default training setting, for quick start.              |
+-------------------------+-------------------+----------------+---------------------------------------------------------+
| `PGDrive-10envs-v0`     | [1000, 1100)      | 10             | Training environment with 10 maps.                      |
+-------------------------+-------------------+----------------+---------------------------------------------------------+
| `PGDrive-1000envs-v0`   | [1000, 1100)      | 1000           | Training environment with 1000 maps.                    |
+-------------------------+-------------------+----------------+---------------------------------------------------------+
| `PGDrive-training0-v0`  | [3000, 4000)      | 1000           | First set of 1000 environments.                         |
+-------------------------+-------------------+----------------+---------------------------------------------------------+
| `PGDrive-training1-v0`  | [5000, 6000)      | 1000           | Second set of 1000 environments.                        |
+-------------------------+-------------------+----------------+---------------------------------------------------------+
| `PGDrive-training2-v0`  | [7000, 8000)      | 1000           | Thirds set of 1000 environments.                        |
+-------------------------+-------------------+----------------+---------------------------------------------------------+
| ...                     |                   |                | *More map set can be added in response to the requests* |
+-------------------------+-------------------+----------------+---------------------------------------------------------+


