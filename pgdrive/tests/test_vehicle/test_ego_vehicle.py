import numpy as np
import pytest

from pgdrive import PGDriveEnv
from pgdrive.scene_creator.vehicle.base_vehicle import BaseVehicle
from pgdrive.envs.pgdrive_env import PGDriveEnvV1_DEFAULT_CONFIG


@pytest.fixture
def pg_env():
    env = PGDriveEnv()
    env.reset()
    return env


def _assert_vehicle(vehicle):
    pos = vehicle.position
    assert -200 < pos[0] < 200
    assert -200 < pos[1] < 200
    speed = vehicle.speed
    assert 0 <= speed <= 120
    velocity_direction = vehicle.velocity_direction
    np.testing.assert_almost_equal(np.linalg.norm(velocity_direction), 1.0)
    current_road = vehicle.current_road
    np.testing.assert_almost_equal(vehicle.heading_diff(vehicle.lane), 0.0, decimal=3)


def _get_heading_deg(heading):
    return (heading * 180 / np.pi)


def test_base_vehicle(pg_env):
    pg_world = pg_env.pg_world
    map = pg_env.current_map

    # v_config = BaseVehicle.get_vehicle_config(dict())
    v = BaseVehicle(pg_world, vehicle_config=PGDriveEnvV1_DEFAULT_CONFIG["vehicle_config"])
    v.add_lidar()
    v.add_routing_localization(True)
    v.add_routing_localization(False)
    v.routing_localization.set_force_calculate_lane_index(True)
    v.update_map_info(map)

    for heading in [-1.0, 0.0, 1.0]:
        for pos in [[0., 0.], [-100., -100.], [100., 100.]]:
            v.reset(map, pos=np.array(pos), heading=heading)
            np.testing.assert_almost_equal(_get_heading_deg(v.heading_theta), heading, decimal=3)
            v_pos = v.position
            # v_pos[1] = -v_pos[1], this position is converted to pg_position in reset() now
            assert np.all(v_pos == pos)
            v.update_state()

    for a_x in [-1, 0, 0.5, 1]:
        for a_y in [-1, 0, 0.5, 1]:
            v.prepare_step([a_x, a_y])
            v.set_act([a_x, a_y])
            _assert_vehicle(v)
            v.set_incremental_action([a_x, a_y])
            _assert_vehicle(v)
            state = v.get_state()
            v.set_state(state)
            assert _get_heading_deg(v.heading_theta) == _get_heading_deg(state["heading"])
            np.testing.assert_almost_equal(v.position, state["position"])
            v.projection([a_x, a_y])

    _nan_speed(pg_env)

    v.destroy()
    del v
    pg_env.close()


def _nan_speed(pg_env):
    steering = [-np.nan, -1, 0, 1, np.nan]
    acc_brake = [-np.nan, -1, 0, 1, np.nan]
    pg_env.reset()
    for s in steering:
        for a in acc_brake:
            pg_env.step([s, a])


if __name__ == '__main__':
    pytest.main(["-sv", "test_ego_vehicle.py"])
