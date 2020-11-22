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