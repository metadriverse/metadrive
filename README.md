<br>

![](metadrive/assets/logo-horizon.png)

<br>

# MetaDrive: Composing Diverse Driving Scenarios for Generalizable RL


<div style="text-align: center; width:100%; margin: 0 auto; display: inline-block">
<strong>
[
<a href="https://metadrive-simulator.readthedocs.io">Documentation</a>
|
<a href="https://www.youtube.com/embed/3ziJPqC_-T4">Demo Video</a>
]
</strong>
</div>

<br>

MetaDrive is a driving simulator with the following key features:

- **Compositional**: It supports generating infinite scenes with various road maps and traffic settings for the research of generalizable RL. 
- **Lightweight**: It is easy to install and run. It can run up to 300 FPS on a standard PC.
- **Realistic**: Accurate physics simulation and multiple sensory input including Lidar, RGB images, top-down semantic map and first-person view images. 


## 🛠 Quick Start
Install MetaDrive via:

```bash
git clone https://github.com/decisionforce/metadrive.git
cd metadrive
pip install numpy cython
pip install -e .
```

or

```bash
pip install metadrive-simulator
```

You can verify the installation of MetaDrive via running the testing script:

```bash
# Go to a folder where no sub-folder calls metadrive
python -m metadrive.examples.profile_metadrive
```

Note that please do not run the above command in a folder that has a sub-folder called `./metadrive`.

## 🚕 Examples

Run the following command to launch a simple driving scenario with auto-drive mode on. Press W, A, S, D to drive the vehicle manually.

```bash
python -m metadrive.examples.drive_in_single_agent_env
```
Run the following command to launch a safe driving scenario, which includes more complex obstacles and cost to be yielded. 

```bash
python -m metadrive.examples.drive_in_safe_metadrive_env
```

You can also launch an instance of Multi-Agent scenario as follows

```bash
python -m metadrive.examples.drive_in_multi_agent_env --env roundabout
```

or launch and render in pygame front end 

```bash
python -m metadrive.examples.drive_in_multi_agent_env --pygame_render --env roundabout
```

env argument could be:
- roundabout (default)
- intersection
- tollgate
- bottleneck
- parkinglot
- pgmap

Run the example of procedural generation of a new map as:

```bash
python -m metadrive.examples.procedural_generation
```

*Note that the above four scripts can not be ran in a headless machine.* 
Please refer to the installation guideline in documentation for more information about how to launch runing in a headless machine.

Run the following command to draw the generated maps from procedural generation:

```bash
python -m metadrive.examples.draw_maps
```

To build the RL environment in python script, you can simply code in the OpenAI gym format as:

```python
import metadrive  # Import this package to register the environment!
import gym

env = gym.make("MetaDrive-v0", config=dict(use_render=True))
# env = metadrive.MetaDriveEnv(config=dict(environment_num=100))  # Or build environment from class
env.reset()
for i in range(1000):
    obs, reward, done, info = env.step(env.action_space.sample())  # Use random policy
    env.render()
    if done:
        env.reset()
env.close()
```


## 📦 Predefined environment sets

We define several standard MetaDrive Gym environments, where the user can start training off the shelf:

```python
import gym
import metadrive  # Register the environment

env = gym.make("MetaDrive-v0")
```

The following table presents the names for the predefined environments. 

|&nbsp;  Gym Environment Name   | Random Seed Range | Number of Maps | Comments                                          |
| ----------------------- | ----------------- | -------------- | ------------------------------------------------------- |
| `MetaDrive-test-v0`       | [0, 200)          | 200            | Test set, not change for all experiments.               |
| `MetaDrive-validation-v0` &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;|[200, 1000)|800| Validation set.|
| `MetaDrive-v0`            | [1000, 1100)      | 100            | Default training setting, for quick start.              |
| `MetaDrive-10envs-v0`     | [1000, 1100)      | 10             | Training environment with 10 maps.                      |
| `MetaDrive-1000envs-v0`   | [1000, 1100)      | 1000           | Training environment with 1000 maps.                    |
| `MetaDrive-training0-v0`  | [3000, 4000)      | 1000           | First set of 1000 environments.                         |
| `MetaDrive-training1-v0`  | [5000, 6000)      | 1000           | Second set of 1000 environments.                        |
| `MetaDrive-training2-v0`  | [7000, 8000)      | 1000           | Thirds set of 1000 environments.                        |
| ...                     |                   |                | *More map set and environments will be added* |



## 🏫 Documentations

Find more details in: [MetaDrive](https://metadrive-simulator.readthedocs.io)


## 📎 References

Working in Progress!

[![build](https://github.com/decisionforce/metadrive/workflows/test/badge.svg)](http://github.com/decisionforce/metadrive/actions)
[![codecov](https://codecov.io/gh/decisionforce/metadrive/branch/main/graph/badge.svg?token=1ZYN8L5397)](https://codecov.io/gh/decisionforce/metadrive)
[![Documentation](https://readthedocs.org/projects/metadrive/badge/?version=latest)](https://metadrive.readthedocs.io)
[![GitHub license](https://img.shields.io/github/license/decisionforce/metadrive)](https://github.com/decisionforce/metadrive/blob/main/LICENSE.txt)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/2d6fabe328a644b49e1269497b741057)](https://www.codacy.com/gh/decisionforce/metadrive/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=decisionforce/metadrive&amp;utm_campaign=Badge_Grade)
[![GitHub contributors](https://img.shields.io/github/contributors/decisionforce/metadrive)](https://github.com/decisionforce/metadrive/graphs/contributors)



<iframe width="560" height="315" src="https://www.youtube.com/embed/3ziJPqC_-T4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
 
