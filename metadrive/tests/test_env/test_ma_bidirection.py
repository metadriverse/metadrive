from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy


def test_ma_bidirection_idm(render=False):
    env = MetaDriveEnv(
        {
            "environment_num": 1,
            "traffic_density": 0,
            "start_seed": 22,
            "manual_control": False,
            "use_render": render,
            "map": "yBY",
            "agent_policy": IDMPolicy,
        }
    )
    index = ('1y0_1_', '2B0_0_', 0)
    try:
        o = env.reset()
        env.vehicle.set_velocity([1, 0.1], 10)
        print(env.vehicle.speed)
        pass_test = False
        for s in range(1, 10000):
            o, r, d, info = env.step(env.action_space.sample())
            _, lat = env.vehicle.lane.local_coordinates(env.vehicle.position)
            if abs(lat) > env.vehicle.lane.width / 2 + 0.1 and len(env.vehicle.navigation.current_ref_lanes) == 1:
                raise ValueError("IDM can not pass bidirection block")
            if env.vehicle.lane.index == index and abs(lat) < 0.1:
                pass_test = True
            if d and info["arrive_dest"]:
                break
        assert pass_test
        env.close()
    finally:
        env.close()


if __name__ == "__main__":
    test_ma_bidirection_idm(True)
