import copy
import logging

from metadrive.component.map.pg_map import PGMap
from metadrive.component.road.road import Road
from metadrive.constants import TARGET_VEHICLES, TRAFFIC_VEHICLES, OBJECT_TO_AGENT
from metadrive.engine.engine_utils import get_engine
from metadrive.manager.traffic_manager import TrafficManager


class Replayer:
    def __init__(self, traffic_mgr: TrafficManager, current_map: PGMap, episode_data: dict):
        self.restore_episode_info = episode_data["frame"]
        self.restore_episode_info.reverse()
        self.restore_vehicles = {}
        self.current_map = current_map
        self._recover_vehicles_from_data(traffic_mgr, episode_data)
        self._init_obj_to_agent = self._record_obj_to_agent()

    def _record_obj_to_agent(self):
        frame = self.restore_episode_info[-1]
        return frame[OBJECT_TO_AGENT]

    def _recover_vehicles_from_data(self, traffic_mgr: TrafficManager, episode_data: dict):
        assert isinstance(self.restore_vehicles, dict), "No place to restore vehicles"
        import metadrive.component.vehicle.vehicle_type as v_types
        traffics = episode_data["init_traffic"]
        engine = get_engine()
        for name, config in traffics.items():
            car_type = getattr(v_types, config["type"])
            car = car_type.create_traffic_vehicle_from_config(traffic_mgr, config)
            self.restore_vehicles[name] = car
            car.attach_to_world(engine.pbr_worldNP, engine.physics_world)
        logging.debug("Recover {} Traffic Vehicles".format(len(self.restore_vehicles)))

    def replay_frame(self, target_vehicles, last_step=False):
        assert self.restore_episode_info is not None, "No frame data in episode info"
        if len(self.restore_episode_info) == 0:
            return True
        frame = self.restore_episode_info.pop(-1)
        vehicles_to_remove = []
        for index, state in frame.items():
            if index == TARGET_VEHICLES:
                for t_v_idx, t_v_s in state.items():
                    agent_idx = self._init_obj_to_agent[t_v_idx]
                    vehicle_to_set = target_vehicles[agent_idx]
                    vehicle_to_set.set_state(t_v_s)
                    if vehicle_to_set.navigation.final_road != Road(*t_v_s["destination"]):
                        vehicle_to_set.navigation.set_route(t_v_s["spawn_road"], t_v_s["destination"][-1])
                    vehicle_to_set.after_step()
            elif index == TRAFFIC_VEHICLES:
                for t_v_idx, t_v_s in state.items():
                    vehicle_to_set = self.restore_vehicles[t_v_idx]
                    vehicle_to_set.set_state(t_v_s)
                    if t_v_s["done"] and not vehicle_to_set.enable_respawn:
                        vehicles_to_remove.append(vehicle_to_set)
        if last_step:
            for v in vehicles_to_remove:
                v.destroy()

    def destroy(self):
        for vehicle in self.restore_vehicles.values():
            vehicle.destroy()
        self.current_map.destroy()

    def __del__(self):
        logging.debug("Replay system is destroyed")


class Recorder:
    def __init__(self, map: PGMap, init_traffic_vehicle_states: dict):
        map_data = dict()
        self.episode_info = dict(
            map_config=map.config.get_dict(),
            init_traffic=init_traffic_vehicle_states,
            map_data=copy.deepcopy(map_data),
            spawn_roads=[road.to_json() for road in map.spawn_roads],
            frame=[]
        )

    def record_frame(self, frame_info: dict):
        self.episode_info["frame"].append(frame_info)

    def dump_episode(self):
        return copy.deepcopy(self.episode_info)

    def destroy(self):
        self.episode_info.clear()

    def __del__(self):
        logging.debug("Record system is destroyed")
