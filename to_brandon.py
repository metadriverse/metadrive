"""
To Brandon:

Install MetaDrive with the following command:
    git clone https://github.com/metadriverse/metadrive.git
    cd metadrive
    pip install -e .

Run this script directly. You can press T in the main interface
to switch between manual control and auto-drive mode.
"""

import numpy as np

from metadrive import MetaDriveEnv
from metadrive.constants import HELP_MESSAGE

if __name__ == "__main__":
    config = dict(
        use_render=True,
        manual_control=True,
        num_scenarios=1,
        vehicle_config=dict(show_lidar=False, show_navi_mark=True, show_line_to_navi_mark=True),
        start_seed=0,
        map="X",  # Set the map to a intersection.
    )
    env = MetaDriveEnv(config)
    try:
        o, _ = env.reset(seed=0)
        print(HELP_MESSAGE)
        env.agent.expert_takeover = True
        assert isinstance(o, np.ndarray)
        print("The observation is an numpy array with shape: ", o.shape)
        for i in range(1, 1000000000):
            o, r, tm, tc, info = env.step([0, 0])
            env.render(
                text={
                    "Auto-Drive (Switch mode: T)": "on" if env.current_track_agent.expert_takeover else "off",
                    "Keyboard Control": "W,A,S,D",
                }
            )
            env.render(mode="topdown")
            if (tm or tc) and info["arrive_dest"]:
                env.reset()
                env.current_track_agent.expert_takeover = True
    finally:
        env.close()
