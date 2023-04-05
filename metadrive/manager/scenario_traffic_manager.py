import copy
import logging

from metadrive.component.traffic_participants.cyclist import Cyclist
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.component.vehicle.vehicle_type import get_vehicle_type
from metadrive.constants import DEFAULT_AGENT
from metadrive.manager.base_manager import BaseManager
from metadrive.policy.replay_policy import ReplayTrafficParticipantPolicy
from metadrive.scenario.parse_object_state import parse_object_state
from metadrive.scenario.scenario_description import ScenarioDescription as SD
from metadrive.type import MetaDriveType

logger = logging.getLogger(__name__)


class ScenarioTrafficManager(BaseManager):
    def __init__(self):
        super(ScenarioTrafficManager, self).__init__()
        self.scenario_id_to_obj_id = None
        self.obj_id_to_scenario_id = None

    def after_reset(self):
        self.scenario_id_to_obj_id = {self.sdc_track_index: self.engine.agent_manager.agent_to_object(DEFAULT_AGENT)}
        self.obj_id_to_scenario_id = {self.engine.agent_manager.agent_to_object(DEFAULT_AGENT): self.sdc_track_index}
        for scenario_id, track in self.current_traffic_data.items():
            if scenario_id == self.sdc_track_index:
                continue
            if track["type"] == MetaDriveType.VEHICLE:
                self.spawn_vehicle(scenario_id, track)
            elif track["type"] == MetaDriveType.CYCLIST:
                self.spawn_cyclist(scenario_id, track)
            elif track["type"] == MetaDriveType.PEDESTRIAN:
                self.spawn_pedestrian(scenario_id, track)
            else:
                logger.info("Do not support {}".format(track["type"]))

    def after_step(self, *args, **kwargs):
        if self.episode_step >= self.scenario_length:
            return dict(default_agent=dict(replay_done=True))

        vehicles_to_clean = []
        for scenario_id, track in self.current_traffic_data.items():
            if scenario_id == self.sdc_track_index:
                continue
            if scenario_id not in self.scenario_id_to_obj_id:
                if track["type"] == MetaDriveType.VEHICLE:
                    self.spawn_vehicle(scenario_id, track)
                elif track["type"] == MetaDriveType.CYCLIST:
                    self.spawn_cyclist(scenario_id, track)
                elif track["type"] == MetaDriveType.PEDESTRIAN:
                    self.spawn_pedestrian(scenario_id, track)
                else:
                    logger.info("Do not support {}".format(track["type"]))
            else:
                policy = self.get_policy(self.scenario_id_to_obj_id[scenario_id])
                if policy.is_current_step_valid:
                    policy.act()
                    # TODO LQY: when using IDM policy, consider add after_step_call
                    # policy.control_object.after_step()
                else:
                    vehicles_to_clean.append(scenario_id)

        for scenario_id in list(vehicles_to_clean):
            obj_id = self.scenario_id_to_obj_id.pop(scenario_id)
            _scenario_id = self.obj_id_to_scenario_id.pop(obj_id)
            assert _scenario_id == scenario_id
            self.clear_objects([obj_id])

        return dict(default_agent=dict(replay_done=False))

    @property
    def current_traffic_data(self):
        return self.engine.data_manager.get_scenario(self.engine.global_random_seed)["tracks"]

    @property
    def sdc_track_index(self):
        return str(self.engine.data_manager.get_scenario(self.engine.global_random_seed)[SD.METADATA][SD.SDC_ID])

    @property
    def scenario_length(self):
        return self.engine.data_manager.get_scenario(self.engine.global_random_seed)[SD.LENGTH]

    @property
    def coordinate_transform(self):
        return self.engine.data_manager.coordinate_transform

    def spawn_vehicle(self, v_id, track):
        state = parse_object_state(track, self.episode_step, self.coordinate_transform)
        if not state["valid"]:
            return
        v_config = copy.deepcopy(self.engine.global_config["vehicle_config"])
        v_config["need_navigation"] = False
        v_config.update(
            dict(
                show_navi_mark=False,
                show_dest_mark=False,
                enable_reverse=False,
                show_lidar=False,
                show_lane_line_detector=False,
                show_side_detector=False,
            )
        )
        if state["vehicle_class"]:
            vehicle_class = state["vehicle_class"]
        else:
            vehicle_class = get_vehicle_type(float(state["length"]), self.np_random)
        obj_name = v_id if self.engine.global_config["force_reuse_object_name"] else None
        v = self.spawn_object(
            vehicle_class, position=state["position"], heading=state["heading"], vehicle_config=v_config, name=obj_name
        )
        self.scenario_id_to_obj_id[v_id] = v.name
        self.obj_id_to_scenario_id[v.name] = v_id
        if self.engine.global_config["replay"]:
            policy = self.add_policy(v.name, ReplayTrafficParticipantPolicy, v, track)
        else:
            raise ValueError("Do not support IDM policy currently")
        policy.act()

    def spawn_pedestrian(self, scenario_id, track):
        state = parse_object_state(track, self.episode_step, self.coordinate_transform)
        if not state["valid"]:
            return
        obj = self.spawn_object(
            Pedestrian,
            position=state["position"],
            heading_theta=state["heading"],
        )
        self.scenario_id_to_obj_id[scenario_id] = obj.name
        self.obj_id_to_scenario_id[obj.name] = scenario_id
        policy = self.add_policy(obj.name, ReplayTrafficParticipantPolicy, obj, track)
        policy.act()

    def spawn_cyclist(self, scenario_id, track):
        state = parse_object_state(track, self.episode_step, self.coordinate_transform)
        if not state["valid"]:
            return
        obj = self.spawn_object(
            Cyclist,
            position=state["position"],
            heading_theta=state["heading"],
        )
        self.scenario_id_to_obj_id[scenario_id] = obj.name
        self.obj_id_to_scenario_id[obj.name] = scenario_id
        policy = self.add_policy(obj.name, ReplayTrafficParticipantPolicy, obj, track)
        policy.act()

    def get_state(self):
        # Record mapping from original_id to new_id
        ret = {}
        ret[SD.ORIGINAL_ID_TO_OBJ_ID] = copy.deepcopy(self.scenario_id_to_obj_id)
        ret[SD.OBJ_ID_TO_ORIGINAL_ID] = copy.deepcopy(self.obj_id_to_scenario_id)
        return ret
