from pgdrive import PGDriveEnv
from pgdrive.constants import TerminationState
from pgdrive.scene_creator.vehicle.traffic_vehicle_type import LVehicle


class SidePassEnv(PGDriveEnv):
    """
    now for test use and demo use only
    """
    @classmethod
    def default_config(cls):
        config = super(SidePassEnv, cls).default_config()
        config.update(
            {
                "environment_num": 1,
                "traffic_density": 0.1,
                "start_seed": 5,
                # "traffic_mode":"respawn",
                "pg_world_config": {
                    "debug_physics_world": False,
                },
                "debug": False,
                "map": "CSRCR"
            }
        )
        return config

    def __init__(self, config=None):
        super(SidePassEnv, self).__init__(config)
        self.breakdown_vehicle = None
        self.alert = None

    def reset(self, episode_data: dict = None):
        ret = super(SidePassEnv, self).reset(episode_data)
        self.vehicle.max_speed = 60
        lane = self.current_map.road_network.graph[">>>"]["1C0_0_"][0]
        self.breakdown_vehicle = self.pgdrive_engine.traffic_manager.spawn_one_vehicle(
            self.pgdrive_engine.traffic_manager.random_vehicle_type(), lane, 30, False
        )
        self.breakdown_vehicle.attach_to_pg_world(self.pgdrive_engine.pbr_worldNP, self.pgdrive_engine.physics_world)

        lane_ = self.current_map.road_network.graph[">>>"]["1C0_0_"][1]
        breakdown_vehicle = self.pgdrive_engine.traffic_manager.spawn_one_vehicle(LVehicle, lane_, 30, False)
        breakdown_vehicle.attach_to_pg_world(self.pgdrive_engine.pbr_worldNP, self.pgdrive_engine.physics_world)

        self.alert = self.pgdrive_engine.object_manager.spawn_one_object(
            "Traffic Triangle", lane, (">>>", "1C0_0_", 0), 22, 0
        )
        self.alert.attach_to_pg_world(self.pgdrive_engine.pbr_worldNP, self.pgdrive_engine.physics_world)

        # part 1
        lane = self.current_map.road_network.graph["1C0_1_"]["2S0_0_"][2]
        pos = [
            (-20, lane.width / 3), (-15.6, lane.width / 4), (-12.1, 0), (-8.7, -lane.width / 4),
            (-4.2, -lane.width / 2), (-0.7, -lane.width * 3 / 4), (4.1, -lane.width), (7.3, -lane.width),
            (11.5, -lane.width), (15.5, -lane.width), (20.0, -lane.width), (23.2, -lane.width), (29.1, -lane.width),
            (32.9, -lane.width / 2), (37.0, 0), (40.0, lane.width / 2)
        ]

        for p in pos:
            cone = self.pgdrive_engine.object_manager.spawn_one_object(
                "Traffic Cone", lane, ("1C0_1_", "2S0_0_", 2), p[0], p[1] * 2 / 3
            )
            cone.attach_to_pg_world(self.pgdrive_engine.pbr_worldNP, self.pgdrive_engine.physics_world)
        from pgdrive.scene_creator.vehicle.traffic_vehicle_type import SVehicle, XLVehicle
        v_pos = [8, 14]
        v_type = [SVehicle, XLVehicle]
        for v_long, v_t in zip(v_pos, v_type):
            v = self.pgdrive_engine.traffic_manager.spawn_one_vehicle(v_t, lane, v_long, False)
            v.attach_to_pg_world(self.pgdrive_engine.pbr_worldNP, self.pgdrive_engine.physics_world)

        # part 2
        lane = self.current_map.road_network.graph["3R0_0_"]["3R0_1_"][0]
        pos = [
            (-20, lane.width / 3), (-15.6, lane.width / 4), (-12.1, 0), (-8.7, -lane.width / 4),
            (-4.2, -lane.width / 2), (-0.7, -lane.width * 3 / 4), (4.1, -lane.width), (7.3, -lane.width),
            (11.5, -lane.width), (15.5, -lane.width), (20.0, -lane.width), (23.2, -lane.width), (29.1, -lane.width),
            (32.9, -lane.width / 2), (37.0, 0), (40.0, lane.width / 2)
        ]

        for p in pos:
            p_ = (p[0] + 5, -p[1])
            cone = self.pgdrive_engine.object_manager.spawn_one_object(
                "Traffic Cone", lane, ("3R0_0_", "3R0_1_", 0), *p_
            )
            cone.attach_to_pg_world(self.pgdrive_engine.pbr_worldNP, self.pgdrive_engine.physics_world)

        v_pos = [14, 19]
        for v_long in v_pos:
            v = self.pgdrive_engine.traffic_manager.spawn_one_vehicle(
                self.pgdrive_engine.traffic_manager.random_vehicle_type(), lane, v_long, False
            )
            v.attach_to_pg_world(self.pgdrive_engine.pbr_worldNP, self.pgdrive_engine.physics_world)

        alert = self.pgdrive_engine.object_manager.spawn_one_object(
            "Traffic Triangle", lane, ("3R0_0_", "3R0_1_", 0), -35, 0
        )
        alert.attach_to_pg_world(self.pgdrive_engine.pbr_worldNP, self.pgdrive_engine.physics_world)

        alert = self.pgdrive_engine.object_manager.spawn_one_object(
            "Traffic Triangle", lane, ("3R0_0_", "3R0_1_", 0), -60, 0
        )
        alert.attach_to_pg_world(self.pgdrive_engine.pbr_worldNP, self.pgdrive_engine.physics_world)

        # part 3
        lane = self.current_map.road_network.graph["4C0_0_"]["4C0_1_"][2]
        pos = [
            (-12.1, 0), (-8.7, -lane.width / 4), (-4.2, -lane.width / 2), (-0.7, -lane.width * 3 / 4),
            (4.1, -lane.width), (7.3, -lane.width), (11.5, -lane.width), (15.5, -lane.width), (20.0, -lane.width),
            (23.2, -lane.width), (29.1, -lane.width), (32.9, -lane.width / 2), (37.0, 0), (40.0, lane.width / 2)
        ]

        for p in pos:
            p_ = (p[0] + 5, p[1] * 3.5 / 3)
            cone = self.pgdrive_engine.object_manager.spawn_one_object(
                "Traffic Cone", lane, ("4C0_0_", "4C0_1_", 2), *p_
            )
            cone.attach_to_pg_world(self.pgdrive_engine.pbr_worldNP, self.pgdrive_engine.physics_world)

        v_pos = [14, 19]
        for v_long in v_pos:
            v = self.pgdrive_engine.traffic_manager.spawn_one_vehicle(
                self.pgdrive_engine.traffic_manager.random_vehicle_type(), lane, v_long, False
            )
            v.attach_to_pg_world(self.pgdrive_engine.pbr_worldNP, self.pgdrive_engine.physics_world)

        # part 4
        lane = self.current_map.road_network.graph["4C0_1_"]["5R0_0_"][0]
        pos = [(-12, lane.width / 4), (-8.1, 0), (-4, -lane.width / 4), (-0.1, -lane.width / 2), (4, -lane.width)]

        for p in pos:
            p_ = (p[0] + 60, -p[1] * 3.5 / 3)
            cone = self.pgdrive_engine.object_manager.spawn_one_object(
                "Traffic Cone", lane, ("4C0_1_", "5R0_0_", 0), *p_
            )
            cone.attach_to_pg_world(self.pgdrive_engine.pbr_worldNP, self.pgdrive_engine.physics_world)

        return ret


if __name__ == "__main__":
    env = SidePassEnv({
        "manual_control": True,
        "use_render": True,
        "vehicle_config": {
            "show_navi_mark": False,
        }
    })

    o = env.reset()
    total_cost = 0
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        total_cost += 1 if info[TerminationState.CRASH_OBJECT] else 0
        env.render(text={"cost": total_cost})
        if d:
            # total_cost = 0
            print("Reset")
            # env.reset()
    env.close()
