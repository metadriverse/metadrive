
<img align=right width=300px  src="pgdrive/assets/PGDrive.png" />

# PGDrive: A highly flexible open-ended driving simulator

Please install PGDrive via:

```bash
pip install git+https://github.com/decisionforce/pgdrive.git
```

or you can install via:

```bash
git clone https://github.com/decisionforce/pgdrive.git
cd pgdrive
pip install -e .
```

## Quick Start

Please run the following line to play with the environment with keyboard!

```bash
python -m pgdrive.examples.enjoy
```

You can also enjoy a journey carrying out by our professional driver! The provided expert can drive in 10000 maps 
with almost 90% likelihood to achieve the destination. 

Note that this script requires your system to have the capacity of rendering. Please refer to the installation guideline for more information.

```bash
python -m pgdrive.examples.enjoy_journey
```

*Note that the above two scripts can not be run in headless machine.*

Running the following line allows you to draw the generated maps:

```bash
python -m pgdrive.examples.draw_maps
```

To build the environment in python script, you can simply run:

```python
import pgdrive  # Import this package to register the environment!
import gym

env = gym.make("PGDrive-v0", config=dict(use_render=True))
env.reset()
for i in range(1000):
    obs, reward, done, info = env.step(env.action_space.sample())  # Use random policy
    env.render()
    if done:
        env.reset()
env.close()
```

We also prepare a Colab notebook which demonstrates some basic usage of PGDrive, please enjoy it! 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decisionforce/PGDrive/blob/main/pgdrive/examples/Basic%20PGDrive%20Usages.ipynb)
(TODO: The notebook is not accessible before the repo is public.)

## Predefined environment sets

We also define several Gym environment names, so user can start training in the minimalist manner:

```python
import gym
import pgdrive  # Register the environment
env = gym.make("PGDrive-v0")
```

The following table presents some predefined environment names. Please feel free to open an issue if you want to request some new environments.

| Gym Environment Name   | Random Seed Range | Number of Maps | Comments                                                |
| ----------------------- | ----------------- | -------------- | ------------------------------------------------------- |
| `PGDrive-test-v0`       | [0, 200)          | 200            | Test set, not change for all experiments.               |
| `PGDrive-validation-v0` | [200, 1000)       | 800            | Validation set.                                         |
| `PGDrive-v0`            | [1000, 1100)      | 100            | Default training setting, for quick start.              |
| `PGDrive-10envs-v0`            | [1000, 1100)      | 10            | Training environment with 10 maps.              |
| `PGDrive-1000envs-v0`            | [1000, 1100)      | 1000            | Training environment with 1000 maps.              |
| `PGDrive-training0-v0`  | [3000, 4000)      | 1000           | First set of 1000 environments.                         |
| `PGDrive-training1-v0`  | [5000, 6000)      | 1000           | Second set of 1000 environments.                        |
| `PGDrive-training2-v0`  | [7000, 8000)      | 1000           | Thirds set of 1000 environments.                        |
| ...                          |                   |                | *More map set can be added in response to the requests* |
