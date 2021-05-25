import copy
import logging

from pgdrive.constants import DEFAULT_AGENT
from pgdrive.scene_creator.map import PGMap
from pgdrive.scene_manager.traffic_manager import TrafficManager
from pgdrive.world.pg_world import PGWorld
from pgdrive.constants import TARGET_VEHICLES, TRAFFIC_VEHICLES, OBJECT_TO_AGENT


class PGReplayer:
    def __init__(self, traffic_mgr: TrafficManager, current_map: PGMap, episode_data: dict, pg_world: PGWorld):
        self.restore_episode_info = episode_data["frame"]
        self.restore_episode_info.reverse()
        self.restore_vehicles = {}
        self.current_map = current_map
        self._recover_vehicles_from_data(traffic_mgr, episode_data, pg_world)
        self._init_obj_to_agent = self._record_obj_to_agent()

    def _record_obj_to_agent(self):
        frame = self.restore_episode_info[-1]
        return frame[OBJECT_TO_AGENT]

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

    def replay_frame(self, target_vehicles, pg_world: PGWorld, last_step=False):
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
            elif index == TRAFFIC_VEHICLES:
                for t_v_idx, t_v_s in state.items():
                    vehicle_to_set = self.restore_vehicles[t_v_idx]
                    vehicle_to_set.set_state(t_v_s)
                    if t_v_s["done"] and not vehicle_to_set.enable_respawn:
                        vehicles_to_remove.append(vehicle_to_set)
        if last_step:
            for v in vehicles_to_remove:
                v.destroy(pg_world)

    def destroy(self, pg_world):
        for vehicle in self.restore_vehicles.values():
            vehicle.destroy(pg_world)
        self.current_map.destroy(pg_world)

    def __del__(self):
        logging.debug("Replay system is destroyed")


class PGRecorder:
    def __init__(self, map: PGMap, init_traffic_vehicle_states: dict):
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
