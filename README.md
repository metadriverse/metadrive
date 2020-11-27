# PG-Drive: An high flexible open-ended driving simulator

Please install PG-Drive via:

```bash
pip install git+https://github.com/decisionforce/pg-drive.git@pre-release
```

or 

```bash
git clone https://github.com/decisionforce/pg-drive.git
cd pg-drive
git checkout pre-release
pip install -e .
```
## Run on cluster

Panda3d needs to be compiled from the source code to turn off the X11 support.

Use the following command to build the panda3d:

```python
python ./makepanda/makepanda.py --no-x11 --everything --no-egl --no-gles --no-gles2 --no-opencv --python-incdir your/path/to/python/include/ --python-libdir your/path/to/python/lib/ --wheel
```

It will generate a .whl file, which can be installed by pip.

## Quick Start

Please run `python -m pg_drive.examples.test_generalization_env` to play with the environment with keyboard!

To build the environment, you can simply run:

```python
import pg_drive  # Import this package to register the environment!
import gym

env = gym.make("GeneralizationRacing-v0", config=dict(use_render=True))
env.reset()
for i in range(1000):
    obs, reward, done, info = env.step(env.action_space.sample())
    env.render()
    if done:
        env.reset()
env.close()
```
