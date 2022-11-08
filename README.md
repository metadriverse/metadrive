<br>

![](metadrive/assets/logo-horizon.png)

<br>

# MetaDrive: Composing Diverse Driving Scenarios for Generalizable RL


<div style="text-align: center; width:100%; margin: 0 auto; display: inline-block">
<strong>
[
<a href="https://metadrive-simulator.readthedocs.io">Documentation</a>
|
<a href="https://colab.research.google.com/github/metadriverse/metadrive/blob/main/metadrive/examples/Basic_MetaDrive_Usages.ipynb">Colab Examples</a>
|
<a href="https://www.youtube.com/embed/3ziJPqC_-T4">Demo Video</a>
|
<a href="https://metadriverse.github.io/metadrive/">Website</a>
|
<a href="https://arxiv.org/pdf/2109.12674.pdf">Paper</a>
|
<a href="https://metadriverse.github.io/">Relevant Projects</a>
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
git clone https://github.com/metadriverse/metadrive.git
cd metadrive
pip install -e .
```

or

```bash
pip install metadrive-simulator
```
*Note that the program is tested on both Linux and Windows. Some control and display issues in MacOS wait to be solved*

You can verify the installation of MetaDrive via running the testing script:

```bash
# Go to a folder where no sub-folder calls metadrive
python -m metadrive.examples.profile_metadrive
```

*Note that please do not run the above command in a folder that has a sub-folder called `./metadrive`.*

## 🚕 Examples
We provide examples to demonstrate features and basic usages of MetaDrive after the local installation.
Or you can run some examples directly in Colab. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/metadriverse/metadrive/blob/main/metadrive/examples/Basic_MetaDrive_Usages.ipynb) 

### Single Agent Environment
Run the following command to launch a simple driving scenario with auto-drive mode on. Press W, A, S, D to drive the vehicle manually.
```bash
python -m metadrive.examples.drive_in_single_agent_env
```
Run the following command to launch a safe driving scenario, which includes more complex obstacles and cost to be yielded. 

```bash
python -m metadrive.examples.drive_in_safe_metadrive_env
```

### Multi-Agent Environment

You can also launch an instance of Multi-Agent scenario as follows

```bash
python -m metadrive.examples.drive_in_multi_agent_env --env roundabout
```
```--env```  accepts following parmeters: `roundabout` (default), `intersection`, `tollgate`, `bottleneck`, `parkinglot`, `pgmap`.
Adding ```--pygame_render``` can launch top-down pygame renderer. 




### Real Environment
Running the following script enables driving in a scenario constructed from Waymo motion dataset.

```bash
python -m metadrive.examples.drive_in_waymo_env
```

Traffic vehicles can not response to surrounding vchicles if directly replaying them.
Add argument ```--reactive_traffic``` to use an IDM policy control them and make them reactive.
Press key ```r``` for loading a new scenario, and ```b``` or ```q``` for switching perspective. 

[comment]: <> (### LQY: avoid introducing these trivial things )

[comment]: <> (Run the example of procedural generation of a new map as:)

[comment]: <> (```bash)

[comment]: <> (python -m metadrive.examples.procedural_generation)

[comment]: <> (```)

[comment]: <> (*Note that the scripts above can not be run in a headless machine.*)

[comment]: <> (*Please refer to the installation guideline in documentation for more information about how to launch runing in a headless machine.*)

[comment]: <> (Run the following command to draw the generated maps from procedural generation:)

[comment]: <> (```bash)

[comment]: <> (python -m metadrive.examples.draw_maps)

[comment]: <> (```)

### Basic Usage
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


## 🏫 Documentations

Find more details in: [MetaDrive](https://metadrive-simulator.readthedocs.io)


## 📎 References

If you use MetaDrive in your own work, please cite:

```latex
@article{li2022metadrive,
  title={Metadrive: Composing diverse driving scenarios for generalizable reinforcement learning},
  author={Li, Quanyi and Peng, Zhenghao and Feng, Lan and Zhang, Qihang and Xue, Zhenghai and Zhou, Bolei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022}
}
```

## 🎉 Relevant Projects

**Learning to Simulate Self-driven Particles System with Coordinated Policy Optimization**
\
Zhenghao Peng, Quanyi Li, Chunxiao Liu, Bolei Zhou 
\
*NeurIPS 2021*
\
[<a href="https://arxiv.org/pdf/2110.13827.pdf" target="_blank">Paper</a>]
[<a href="https://github.com/decisionforce/CoPO" target="_blank">Code</a>]
[<a href="https://decisionforce.github.io/CoPO" target="_blank">Webpage</a>]
[<a href="https://decisionforce.github.io/CoPO/copo_poster.pdf" target="_blank">Poster</a>]
[<a href="https://youtu.be/sOw43l8lwxE" target="_blank">Talk</a>]
[<a href="https://github.com/metadriverse/metadrive-benchmark/tree/main/MARL" target="_blank">Results&Models</a>]


**Safe Driving via Expert Guided Policy Optimization**
\
Zhenghao Peng*, Quanyi Li*, Chunxiao Liu, Bolei Zhou
\
*Conference on Robot Learning (CoRL) 2021*
\
[<a href="https://arxiv.org/pdf/2110.06831.pdf" target="_blank">Paper</a>]
[<a href="https://github.com/decisionforce/EGPO" target="_blank">Code</a>]
[<a href="https://decisionforce.github.io/EGPO/" target="_blank">Webpage</a>]
[<a href="https://decisionforce.github.io/EGPO/images/egpo_poster.png" target="_blank">Poster</a>]

**Efficient Learning of Safe Driving Policy via Human-AI Copilot Optimization**
\
Quanyi Li*, Zhenghao Peng*, Bolei Zhou
\
*ICLR 2022*
\
[<a href="https://arxiv.org/pdf/2202.10341.pdf" target="_blank">Paper</a>]
[<a href="https://github.com/decisionforce/HACO" target="_blank">Code</a>]
[<a href="https://decisionforce.github.io/HACO/" target="_blank">Webpage</a>]
[<a href="https://github.com/decisionforce/HACO/blob/main/docs/iclr_poster.pdf" target="_blank">Poster</a>]
[<a href="https://youtu.be/PiJv4wtp8T8" target="_blank">Talk</a>]

**Human-AI Shared Control via Policy Dissection**
\
Quanyi Li, Zhenghao Peng, Haibin Wu, Lan Feng, Bolei Zhou
\
*NeurIPS 2022*
\
[<a href="https://arxiv.org/pdf/2206.00152.pdf" target="_blank">Paper</a>]
[<a href="https://github.com/metadriverse/policydissect" target="_blank">Code</a>]
[<a href="https://metadriverse.github.io/policydissect/" target="_blank">Webpage</a>]

[![build](https://github.com/metadriverse/metadrive/workflows/test/badge.svg)](http://github.com/metadriverse/metadrive/actions)
[![Documentation](https://readthedocs.org/projects/metadrive/badge/?version=latest)](https://metadrive.readthedocs.io)
[![GitHub license](https://img.shields.io/github/license/metadriverse/metadrive)](https://github.com/metadriverse/metadrive/blob/main/LICENSE.txt)
[![GitHub contributors](https://img.shields.io/github/contributors/metadriverse/metadrive)](https://github.com/metadriverse/metadrive/graphs/contributors)
