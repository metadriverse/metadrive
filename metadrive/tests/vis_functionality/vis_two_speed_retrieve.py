from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils import setup_logger

setup_logger(debug=True)

if __name__ == "__main__":
    import numpy as np

    env = MetaDriveEnv(
        {
            "num_scenarios": 4,
            "traffic_density": 0.0,
            "use_render": True,
            "map_config": {
                BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                BaseMap.GENERATE_CONFIG: "SSSSSSSSSSSSS",
            },
            "manual_control": True
        }
    )
    acc = [0, 1]
    brake = [-1, -np.nan]
    env.reset()
    for i in range(1, 100000):
        o, r, tm, tc, info = env.step(acc)
        # print(
        #     "new:{}, old:{}, diff:{}".format(
        #         env.agent.speed_km_h, env.agent.system.get_current_speed_km_hour(),
        #         env.agent.speed_km_h - env.agent.system.get_current_speed_km_hour()
        #     )
        # )
        env.render("Test: {}".format(i))
    env.close()
