from metadrive.component.static_object.traffic_object import TrafficCone, TrafficWarning
from metadrive.component.vehicle.vehicle_type import LVehicle
from metadrive.constants import TerminationState, DEFAULT_AGENT
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.manager.object_manager import TrafficObjectManager


class ComplexObjectManager(TrafficObjectManager):
    def reset(self):
        ret = super(ComplexObjectManager, self).reset()
        lane = self.engine.current_map.road_network.graph[">>>"]["1C0_0_"][0]
        breakdown_vehicle = self.engine.object_manager.spawn_object(
            self.engine.traffic_manager.random_vehicle_type(),
            vehicle_config={
                "spawn_lane_index": lane.index,
                "spawn_longitude": 30
            }
        )
        self.engine.object_manager.accident_lanes.append(lane)
        lane_ = self.engine.current_map.road_network.graph[">>>"]["1C0_0_"][1]
        breakdown_vehicle = self.engine.object_manager.spawn_object(
            LVehicle, vehicle_config={
                "spawn_lane_index": lane_.index,
                "spawn_longitude": 30
            }
        )
        self.engine.object_manager.accident_lanes.append(lane_)
        lane = lane
        longitude = 22
        lateral = 0
        alert = self.engine.object_manager.spawn_object(
            TrafficWarning,
            lane=lane,
            position=lane.position(longitude, lateral),
            heading_theta=lane.heading_theta_at(longitude),
        )

        # part 1
        lane = self.engine.current_map.road_network.graph["1C0_1_"]["2S0_0_"][2]
        pos = [
            (-20, lane.width / 3), (-15.6, lane.width / 4), (-12.1, 0), (-8.7, -lane.width / 4),
            (-4.2, -lane.width / 2), (-0.7, -lane.width * 3 / 4), (4.1, -lane.width), (7.3, -lane.width),
            (11.5, -lane.width), (15.5, -lane.width), (20.0, -lane.width), (23.2, -lane.width), (29.1, -lane.width),
            (32.9, -lane.width / 2), (37.0, 0), (40.0, lane.width / 2)
        ]

        for p in pos:
            longitude = p[0]
            lateral = p[1] + lane.width / 2
            cone = self.engine.object_manager.spawn_object(
                TrafficCone,
                lane=lane,
                position=lane.position(longitude, lateral),
                heading_theta=lane.heading_theta_at(longitude)
            )
        self.engine.object_manager.accident_lanes.append(lane)
        from metadrive.component.vehicle.vehicle_type import SVehicle, XLVehicle
        v_pos = [8, 14]
        v_type = [SVehicle, XLVehicle]
        for v_long, v_t in zip(v_pos, v_type):
            v = self.engine.object_manager.spawn_object(
                v_t, vehicle_config={
                    "spawn_lane_index": lane.index,
                    "spawn_longitude": v_long
                }
            )

        # part 2
        lane = self.engine.current_map.road_network.graph["3R0_0_"]["3R0_1_"][0]
        self.engine.object_manager.accident_lanes.append(lane)
        pos = [
            (-20, lane.width / 3), (-15.6, lane.width / 4), (-12.1, 0), (-8.7, -lane.width / 4),
            (-4.2, -lane.width / 2), (-0.7, -lane.width * 3 / 4), (4.1, -lane.width), (7.3, -lane.width),
            (11.5, -lane.width), (15.5, -lane.width), (20.0, -lane.width), (23.2, -lane.width), (29.1, -lane.width),
            (32.9, -lane.width / 2), (37.0, 0), (40.0, lane.width / 2)
        ]

        for p in pos:
            p_ = (p[0] + 5, -p[1])
            longitude = p_[0]
            lateral = -p[1] - lane.width / 2
            cone = self.engine.object_manager.spawn_object(
                TrafficCone,
                lane=lane,
                position=lane.position(longitude, lateral),
                heading_theta=lane.heading_theta_at(longitude)
            )

        v_pos = [14, 19]
        for v_long in v_pos:
            v = self.engine.object_manager.spawn_object(
                self.engine.traffic_manager.random_vehicle_type(),
                vehicle_config={
                    "spawn_lane_index": lane.index,
                    "spawn_longitude": v_long
                }
            )
        longitude = -35
        lateral = 0
        alert = self.engine.object_manager.spawn_object(
            TrafficWarning,
            lane=lane,
            position=lane.position(longitude, lateral),
            heading_theta=lane.heading_theta_at(longitude)
        )
        longitude = -60
        lateral = 0
        alert = self.engine.object_manager.spawn_object(
            TrafficWarning,
            lane=lane,
            position=lane.position(longitude, lateral),
            heading_theta=lane.heading_theta_at(longitude)
        )

        # part 3
        lane = self.engine.current_map.road_network.graph["4C0_0_"]["4C0_1_"][2]
        self.engine.object_manager.accident_lanes.append(lane)
        pos = [
            (-12.1, 0), (-8.7, -lane.width / 4), (-4.2, -lane.width / 2), (-0.7, -lane.width * 3 / 4),
            (4.1, -lane.width), (7.3, -lane.width), (11.5, -lane.width), (15.5, -lane.width), (20.0, -lane.width),
            (23.2, -lane.width), (29.1, -lane.width), (32.9, -lane.width / 2), (37.0, 0), (40.0, lane.width / 2)
        ]

        for p in pos:
            p_ = (p[0] + 5, p[1] * 3.5 / 3)
            longitude = p_[0]
            lateral = p[1] + lane.width / 2
            cone = self.engine.object_manager.spawn_object(
                TrafficCone,
                lane=lane,
                position=lane.position(longitude, lateral),
                heading_theta=lane.heading_theta_at(longitude)
            )

        v_pos = [14, 19]
        for v_long in v_pos:
            v = self.engine.object_manager.spawn_object(
                self.engine.traffic_manager.random_vehicle_type(),
                vehicle_config={
                    "spawn_lane_index": lane.index,
                    "spawn_longitude": v_long
                }
            )

        # part 4
        lane = self.engine.current_map.road_network.graph["4C0_1_"]["5R0_0_"][0]
        self.engine.object_manager.accident_lanes.append(lane)
        # pos = [(-12, lane.width / 4), (-8.1, 0), (-4, -lane.width / 4), (-0.1, -lane.width / 2), (4, -lane.width)]
        #
        # for p in pos:
        #     p_ = (p[0] + 60, -p[1] * 3.5 / 3)
        #     cone = self.engine.object_manager.spawn_object(TrafficCone, lane=lane, longitude=p_[0], lateral=p_[1])

        return ret


class ComplexEnv(SafeMetaDriveEnv):
    """
    now for test use and demo use only
    """
    def default_config(self):
        config = super(ComplexEnv, self).default_config()
        config.update(
            {
                "num_scenarios": 1,
                "traffic_density": 0.05,
                "start_seed": 5,
                "accident_prob": 0.0,
                # "traffic_mode":"respawn",
                "debug_physics_world": False,
                "debug": False,
                "map": "CSRCR"
            }
        )
        return config

    def setup_engine(self):
        super(ComplexEnv, self).setup_engine()
        self.engine.register_manager("object_manager", ComplexObjectManager())


def test_object_collision_detection(render=False):
    env = ComplexEnv(
        {
            # "manual_control": True,
            "traffic_density": 0.0,
            "use_render": render,
            "crash_object_cost": 100,
            "crash_object_done": True,
            "debug": False,
            "vehicle_config": {
                "show_lidar": True
            }
        }
    )
    try:
        o, _ = env.reset()
        lane_index = (">>", ">>>", 0)
        lane = env.current_map.road_network.get_lane(lane_index)
        longitude = 22
        lateral = 0
        alert = env.engine.object_manager.spawn_object(
            TrafficWarning,
            lane=env.current_map.road_network.get_lane(lane_index),
            position=lane.position(longitude, lateral),
            heading_theta=lane.heading_theta_at(longitude)
        )
        lane_index = (">>", ">>>", 2)

        longitude = 22
        lateral = 0
        lane = env.current_map.road_network.get_lane(lane_index)
        alert = env.engine.object_manager.spawn_object(
            TrafficCone,
            lane=env.current_map.road_network.get_lane(lane_index),
            position=lane.position(longitude, lateral),
            heading_theta=lane.heading_theta_at(longitude)
        )
        crash_obj = False
        detect_obj = False
        for i in range(1, 100000 if render else 2000):
            o, r, tm, tc, info = env.step([0, 1])
            for obj in env.observations[DEFAULT_AGENT].detected_objects:
                if isinstance(obj, TrafficCone):
                    detect_obj = True
            if render:
                env.render()
            if info["cost"] == 100 and info[TerminationState.CRASH_OBJECT]:
                crash_obj = True
                break
        assert crash_obj and detect_obj, "Can not crash with object!"
    finally:
        env.close()


if __name__ == "__main__":
    test_object_collision_detection(render=False)
