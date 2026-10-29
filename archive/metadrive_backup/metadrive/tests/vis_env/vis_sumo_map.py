"""use netconvert --opendrive-files CARLA_town01.net.xml first"""
import logging

import numpy as np

from metadrive.component.lane.point_lane import PointLane
from metadrive.component.vehicle.vehicle_type import SVehicle
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs import BaseEnv
from metadrive.manager.base_manager import BaseManager
from metadrive.manager.sumo_map_manager import SumoMapManager
from metadrive.obs.observation_base import DummyObservation
from metadrive.policy.idm_policy import TrajectoryIDMPolicy
from metadrive.utils.pg.utils import ray_localization


class SimpleTrafficManager(BaseManager):
    """
    A simple traffic creator, which creates one vehicle to follow a specified route with IDM policy.
    """
    def __init__(self):
        super(SimpleTrafficManager, self).__init__()
        self.generated_v = None
        self.arrive_dest = False

    def after_reset(self):
        """
        Create vehicle and use IDM for controlling it. When there are objects in front of the vehicle, it will yield
        """
        self.arrive_dest = False
        path_to_follow = []
        for lane_index in ["lane_4_0", "lane_:306_0_0", "lane_22_0"]:
            path_to_follow.append(self.engine.current_map.road_network.get_lane(lane_index).get_polyline())
        path_to_follow = np.concatenate(path_to_follow, axis=0)

        self.generated_v = self.spawn_object(
            SVehicle, vehicle_config=dict(), position=path_to_follow[60], heading=-np.pi
        )
        TrajectoryIDMPolicy.NORMAL_SPEED = 20
        self.add_policy(
            self.generated_v.id,
            TrajectoryIDMPolicy,
            control_object=self.generated_v,
            random_seed=0,
            traj_to_follow=PointLane(path_to_follow, 2)
        )

    def before_step(self):
        """
        When arrive destination, stop
        """
        policy = self.get_policy(self.generated_v.id)
        if policy.arrive_destination:
            self.arrive_dest = True

        if not self.arrive_dest:
            action = policy.act(do_speed_control=True)
        else:
            action = [0., -1]
        self.generated_v.before_step(action)  # set action


class MyEnv(BaseEnv):
    def reward_function(self, agent):
        """Dummy reward function."""
        return 0, {}

    def cost_function(self, agent):
        """Dummy cost function."""
        return 0, {}

    def done_function(self, agent):
        """Dummy done function."""
        return False, {}

    def get_single_observation(self):
        """Dummy observation function."""
        return DummyObservation()

    def setup_engine(self):
        """Register the map manager"""
        super().setup_engine()
        map_path = AssetLoader.file_path("carla", "CARLA_town01.net.xml", unix_style=False)
        self.engine.register_manager("map_manager", SumoMapManager(map_path))
        self.engine.register_manager("traffic_manager", SimpleTrafficManager())


if __name__ == "__main__":
    # create env
    env = MyEnv(
        dict(
            use_render=True,
            vehicle_config={"spawn_position_heading": [(0, 0), np.pi / 2]},
            manual_control=True,  # we usually manually control the car to test environment
            use_mesh_terrain=True,
            log_level=logging.CRITICAL
        )
    )  # suppress logging message
    env.reset()
    for i in range(10000):
        # step
        obs, reward, termination, truncate, info, = env.step(env.action_space.sample())
        current_lane_indices = [
            info[1] for info in
            ray_localization(env.vehicle.heading, env.vehicle.position, env.engine, use_heading_filter=True)
        ]

        env.render(text={"current_lane_indices": current_lane_indices})
    env.close()
