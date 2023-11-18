import numpy as np

from metadrive import MetaDriveEnv
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.envs.base_env import BASE_DEFAULT_CONFIG
from metadrive.envs.metadrive_env import METADRIVE_DEFAULT_CONFIG
from metadrive.utils import Config
from metadrive.utils.coordinates_shift import metadrive_vector


def _assert_vehicle(vehicle):
    pos = vehicle.position
    assert -200 < pos[0] < 200
    assert -200 < pos[1] < 200
    speed = vehicle.speed_km_h
    assert 0 <= speed <= 120
    velocity_direction = vehicle.heading
    np.testing.assert_almost_equal(abs(np.linalg.norm(velocity_direction)), 1.0)
    current_road = vehicle.navigation.current_road
    np.testing.assert_almost_equal(vehicle.heading_diff(vehicle.lane), 0.5, decimal=3)


def _get_heading_deg(heading):
    return (heading * 180 / np.pi)


def test_base_vehicle():
    env = MetaDriveEnv()
    try:
        env.reset()
        engine = env.engine
        map = env.current_map

        # v_config = BaseVehicle.get_vehicle_config(dict())
        v_config = Config(BASE_DEFAULT_CONFIG["vehicle_config"]).update(METADRIVE_DEFAULT_CONFIG["vehicle_config"])
        v_config["spawn_lane_index"] = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0)
        v = engine.spawn_object(DefaultVehicle, vehicle_config=v_config, random_seed=0)

        v.add_navigation()
        v.add_navigation()
        v.navigation.set_force_calculate_lane_index(True)
        v.reset_navigation()

        for heading in [-1.0, 0.0, 1.0]:
            for pos in [[0., 0.], [-100., -100.], [100., 100.]]:
                # v.reset(position=pos, heading=heading)
                # np.testing.assert_almost_equal(_get_heading_deg(v.heading_theta), heading, decimal=3)
                #
                # v_pos = v.position
                # # v_pos[1] = -v_pos[1], this position is converted to pg_position in reset() now
                # np.testing.assert_almost_equal(v_pos, pos)

                v.set_position(pos)
                v_pos = v.position
                np.testing.assert_almost_equal(v_pos, pos)

                v.after_step()
        v.reset(position=np.array([10, 0]))
        for a_x in [-1, 0, 0.5, 1]:
            for a_y in [-1, 0, 0.5, 1]:
                v.before_step([a_x, a_y])
                v._set_action([a_x, a_y])
                _assert_vehicle(v)
                v._set_incremental_action([a_x, a_y])
                _assert_vehicle(v)
                state = v.get_state()
                v.set_state(state)
                assert _get_heading_deg(v.heading_theta) == _get_heading_deg(state["heading_theta"])
                np.testing.assert_almost_equal(v.position, metadrive_vector(state["position"]))
                v.convert_to_local_coordinates([a_x, a_y], 0.0)
        env.engine.clear_objects([v.id])
        _nan_speed(env)
        v = engine.spawn_object(DefaultVehicle, vehicle_config=v_config, random_seed=0)
        v.reset(position=np.array([10, 0]))
        v.set_heading_theta(np.pi / 2)
        bounding_box = np.asarray(v.bounding_box)
        l = v.LENGTH
        w = v.WIDTH
        points = np.asarray([
            [10 - w / 2, l / 2],
            [10 + w / 2, l / 2],
            [10 + w / 2, -l / 2],
            [10 - w / 2, -l / 2],
        ])
        np.testing.assert_almost_equal(bounding_box, points)

        v.destroy()
        del v
    finally:
        env.close()


def _nan_speed(pg_env):
    steering = [-np.nan, -1, 0, 1, np.nan]
    acc_brake = [-np.nan, -1, 0, 1, np.nan]
    pg_env.reset()
    for s in steering:
        for a in acc_brake:
            pg_env.step([s, a])


if __name__ == '__main__':
    # pytest.main(["-sv", "test_ego_vehicle.py"])
    test_base_vehicle()
