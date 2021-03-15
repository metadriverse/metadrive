import copy
import logging

from pgdrive.constants import DEFAULT_AGENT
from pgdrive.scene_creator.map import Map
from pgdrive.scene_manager.traffic_manager import TrafficManager
from pgdrive.world.pg_world import PGWorld


class PGReplayer:
    def __init__(self, traffic_mgr: TrafficManager, current_map: Map, episode_data: dict, pg_world: PGWorld):
        self.restore_episode_info = episode_data["frame"]
        self.restore_episode_info.reverse()
        self.restore_vehicles = {}
        self.current_map = current_map
        self._recover_vehicles_from_data(traffic_mgr, episode_data, pg_world)

    def _recover_vehicles_from_data(self, traffic_mgr: TrafficManager, episode_data: dict, pg_world: PGWorld):
        assert isinstance(self.restore_vehicles, dict), "No place to restore vehicles"
        import pgdrive.scene_creator.vehicle.traffic_vehicle_type as v_types
        traffics = episode_data["init_traffic"]
        for name, config in traffics.items():
            car_type = getattr(v_types, config["type"])
            car = car_type.create_traffic_vehicle_from_config(traffic_mgr, config)
            self.restore_vehicles[name] = car
            car.attach_to_pg_world(pg_world.pbr_worldNP, pg_world.physics_world)
        logging.debug("Recover {} Traffic Vehicles".format(len(self.restore_vehicles)))

    def replay_frame(self, ego_vehicle, pg_world: PGWorld):
        assert self.restore_episode_info is not None, "Not frame data in episode info"
        if len(self.restore_episode_info) == 0:
            return True
        frame = self.restore_episode_info.pop(-1)
        vehicles_to_remove = []
        for index, state in frame.items():
            if index == "ego":
                vehicle_to_set = ego_vehicle
                assert len(state) == 1, "Only support single-agent now!"
                state = state[DEFAULT_AGENT]
                vehicle_to_set.set_state(state)
            else:
                vehicle_to_set = self.restore_vehicles[index]
                vehicle_to_set.set_state(state)
                if state["done"] and not vehicle_to_set.enable_reborn:
                    vehicles_to_remove.append(vehicle_to_set)
        for v in vehicles_to_remove:
            v.destroy(pg_world)

    def destroy(self, pg_world):
        for vehicle in self.restore_vehicles.values():
            vehicle.destroy(pg_world)
        self.current_map.destroy(pg_world)

    def __del__(self):
        logging.debug("Replay system is destroyed")


class PGRecorder:
    def __init__(self, map: Map, init_traffic_vehicle_states: dict):
        map_data = dict()
        map_data[map.random_seed] = map.save_map()
        self.episode_info = dict(
            map_config=map.config.get_dict(),
            init_traffic=init_traffic_vehicle_states,
            map_data=copy.deepcopy(map_data),
            frame=[]
        )

    def record_frame(self, frame_info: dict):
        self.episode_info["frame"].append(frame_info)

    def dump_episode(self):
        return copy.deepcopy(self.episode_info)

    def destroy(self, pg_world: PGWorld):
        self.episode_info.clear()

    def __del__(self):
        logging.debug("Record system is destroyed")
