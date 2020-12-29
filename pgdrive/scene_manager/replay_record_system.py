from pgdrive.world.pg_world import PgWorld
import logging
from . import TrafficMode
import copy
from pgdrive.scene_creator.map import Map


class PGReplayer:
    def __init__(self, scene_mgr, current_map: Map, episode_data: dict, pg_world: PgWorld):
        self.restore_traffic_mode = episode_data["traffic_mode"]
        self.restore_episode_info = episode_data["frame"]
        self.restore_episode_info.reverse()
        self.restore_vehicles = {}
        self.current_map = current_map
        self._recover_vehicles_from_data(scene_mgr, episode_data, pg_world)

    def _recover_vehicles_from_data(self, scene_mgr, episode_data: dict, pg_world: PgWorld):
        assert isinstance(self.restore_vehicles, dict), "No place to restore vehicles"
        import pgdrive.scene_creator.pg_traffic_vehicle.traffic_vehicle_type as v_types
        traffics = episode_data["init_traffic"]
        for name, config in traffics.items():
            car_type = getattr(v_types, config["type"])
            car = car_type.create_traffic_vehicle_from_config(scene_mgr, config)
            self.restore_vehicles[name] = car
            car.attach_to_pg_world(pg_world.pbr_worldNP, pg_world.physics_world)

    def replay_frame(self, ego_vehicle, pg_world: PgWorld):
        assert self.restore_episode_info is not None, "Not frame data in episode info"
        if len(self.restore_episode_info) == 0:
            return True
        frame = self.restore_episode_info.pop(-1)
        vehicles_to_remove = []
        for index, state in frame.items():
            vehicle = self.restore_vehicles[index] if index != "ego" else ego_vehicle
            vehicle.set_state(state)
            if index != "ego" and state["done"] and self.restore_traffic_mode == TrafficMode.Add_once:
                vehicles_to_remove.append(vehicle)
        for v in vehicles_to_remove:
            v.destroy(pg_world)

    def destroy(self, pg_world):
        for vehicle in self.restore_vehicles.values():
            vehicle.destroy(pg_world)
        self.current_map.destroy(pg_world)

    def __del__(self):
        logging.debug("Replay system is destroyed")


class PGRecorder:
    def __init__(self, map: Map, init_vehicle_states: dict, traffic_mode: TrafficMode):
        map_data = dict()
        map_data[map.random_seed] = map.save_map()
        init_vehicle_state = init_vehicle_states
        self.episode_info = dict(
            map_config=map.config.get_dict(),
            init_traffic=init_vehicle_state,
            traffic_mode=traffic_mode,
            map_data=copy.deepcopy(map_data),
            frame=[]
        )

    def record_frame(self, frame_info: dict):
        self.episode_info["frame"].append(frame_info)

    def dump_episode(self):
        return copy.deepcopy(self.episode_info)

    def destroy(self, pg_world: PgWorld):
        self.episode_info.clear()

    def __del__(self):
        logging.debug("Record system is destroyed")
