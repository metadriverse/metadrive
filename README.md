<br>

![](documentation/source/figs/logo-horizon.png)

<br>

# MetaDrive: an Open-source Driving Simulator for AI and Autonomy Research

[![build](https://github.com/metadriverse/metadrive/workflows/test/badge.svg)](http://github.com/metadriverse/metadrive/actions)
[![Documentation](https://readthedocs.org/projects/metadrive-simulator/badge/?version=latest)](https://metadrive-simulator.readthedocs.io)
[![GitHub license](https://img.shields.io/github/license/metadriverse/metadrive)](https://github.com/metadriverse/metadrive/blob/main/LICENSE.txt)
[![GitHub contributors](https://img.shields.io/github/contributors/metadriverse/metadrive)](https://github.com/metadriverse/metadrive/graphs/contributors)
[![Downloads](https://static.pepy.tech/badge/MetaDrive-simulator)](https://pepy.tech/project/MetaDrive-simulator)

<div style="text-align: center; width:100%; margin: 0 auto; display: inline-block">
<strong>
[
<a href="https://metadrive-simulator.readthedocs.io">Documentation</a>
|
<a href="https://github.com/metadriverse/metadrive?tab=readme-ov-file#-examples">Colab Examples</a>
|
<a href="https://www.youtube.com/embed/3ziJPqC_-T4">Demo Video</a>
|
<a href="https://metadriverse.github.io/metadrive-simulator/">Website</a>
|
<a href="https://arxiv.org/pdf/2109.12674.pdf">Paper</a>
|
<a href="https://metadriverse.github.io/">Relevant Projects</a>
]
</strong>
</div>

<br>

MetaDrive is a driving simulator with the following key features:

- **Compositional**: It supports synthesising infinite scenes with various road maps and traffic settings or loading real-world driving logs for the research of generalizable RL. 
- **Lightweight**: It is easy to install and run on Linux/Windows/MacOS with sensor simulation support. It can run up to +1000 FPS on a standard PC.
- **Realistic**: Accurate physics simulation and multiple sensory input including point cloud, RGB/Depth/Semantic images, top-down semantic map and first-person view images. 


## 🛠 Quick Start
Install MetaDrive via:

```bash
git clone https://github.com/metadriverse/metadrive.git
cd metadrive
pip install -e .
```

You can verify the installation of MetaDrive via running the testing script:

```bash
# Go to a folder where no sub-folder calls metadrive
python -m metadrive.examples.profile_metadrive
```

*Note that please do not run the above command in a folder that has a sub-folder called `./metadrive`.*

## 🚕 Examples
We provide [examples](https://github.com/metadriverse/metadrive/tree/main/metadrive/examples) to demonstrate features and basic usages of MetaDrive after the local installation.
There is an `.ipynb` example which can be directly opened in Colab. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/metadriverse/metadrive/blob/main/metadrive/examples/Basic_MetaDrive_Usages.ipynb)

Also, you can try examples in the documentation directly in Colab! See more details in [Documentations](#-documentations).

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
Adding ```--top_down``` can launch top-down pygame renderer. 




### Real Environment
Running the following script enables driving in a scenario constructed from nuScenes dataset or Waymo dataset.

```bash
python -m metadrive.examples.drive_in_real_env
```

The default real-world dataset is nuScenes.
Use ```--waymo``` to visualize Waymo scenarios.
Traffic vehicles can not response to surrounding vchicles if directly replaying them.
Add argument ```--reactive_traffic``` to use an IDM policy control them and make them reactive.
Press key ```r``` for loading a new scenario, and ```b``` or ```q``` for switching perspective. 



### Basic Usage
To build the RL environment in python script, you can simply code in the Farama Gymnasium format as:

```python
from metadrive.envs.metadrive_env import MetaDriveEnv

env = MetaDriveEnv(config={"use_render": True})
obs, info = env.reset()
for i in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        env.reset()
env.close()
```


## 🏫 Documentations

Please find more details in: https://metadrive-simulator.readthedocs.io

### Running Examples in Doc
The documentation is built with `.ipynb` so every example can run locally
or with colab. For Colab running, on the Colab interface, click “GitHub,” enter the URL of MetaDrive:
https://github.com/metadriverse/metadrive, and hit the search icon.
After running examples, you are expected to get the same output and visualization results as the documentation!

For example, hitting the following icon opens the source `.ipynb` file of the documentation section: [Environments](https://metadrive-simulator.readthedocs.io/en/latest/rl_environments.html).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/metadriverse/metadrive/blob/main/documentation/source/rl_environments.ipynb)

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


## Acknowledgement

The simulator can not be built without the help from Panda3D community and the following open-sourced projects:
- panda3d-simplepbr: https://github.com/Moguri/panda3d-simplepbr
- panda3d-gltf: https://github.com/Moguri/panda3d-gltf
- RenderPipeline (RP): https://github.com/tobspr/RenderPipeline
- Water effect for RP: https://github.com/kergalym/RenderPipeline 
- procedural_panda3d_model_primitives: https://github.com/Epihaius/procedural_panda3d_model_primitives
- DiamondSquare for terrain generation: https://github.com/buckinha/DiamondSquare
- KITSUNETSUKI-Asset-Tools: https://github.com/kitsune-ONE-team/KITSUNETSUKI-Asset-Tools

