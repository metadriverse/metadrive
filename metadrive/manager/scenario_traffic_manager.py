import copy
import logging

import numpy as np

from metadrive.component.static_object.traffic_object import TrafficCone, TrafficBarrier
from metadrive.component.traffic_participants.cyclist import Cyclist
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.component.vehicle.vehicle_type import get_vehicle_type, reset_vehicle_type_count
from metadrive.constants import DEFAULT_AGENT
from metadrive.manager.base_manager import BaseManager
from metadrive.policy.idm_policy import ScenarioIDMPolicy
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.policy.replay_policy import ReplayTrafficParticipantPolicy
from metadrive.scenario.parse_object_state import parse_object_state, get_idm_route, get_max_valid_indicis
from metadrive.scenario.scenario_description import ScenarioDescription as SD
from metadrive.type import MetaDriveType
from metadrive.utils.math import wrap_to_pi

logger = logging.getLogger(__name__)


class ScenarioTrafficManager(BaseManager):
    STATIC_THRESHOLD = 3  # m, static if moving distance < 5
    IDM_ACT_BATCH_SIZE = 5

    # project cars to ego vehicle coordinates, only vehicles behind ego car and in a certain region can get IDM policy
    IDM_CREATE_SIDE_CONSTRAINT = 15  # m
    IDM_CREATE_FORWARD_CONSTRAINT = -1  # m
    IDM_CREATE_MIN_LENGTH = 5  # m

    # project cars to ego vehicle coordinates, only vehicles outside the region can be created
    GENERATION_SIDE_CONSTRAINT = 2  # m
    GENERATION_FORWARD_CONSTRAINT = 8  # m

    # filter noise static object: barrier and cone
    MIN_VALID_FRAME_LEN = 20  # frames

    def __init__(self):
        super(ScenarioTrafficManager, self).__init__()
        self._scenario_id_to_obj_id = None
        self._obj_id_to_scenario_id = None

        # for filtering some static cars
        self._static_car_id = set()
        self._moving_car_id = set()

        # for filtering noisy static object
        self._noise_object_id = set()
        self._non_noise_object_id = set()

        # an async trick for accelerating IDM policy
        self.idm_policy_count = 0
        self._obj_to_clean_this_frame = []

        # some flags
        self.even_sample_v = self.engine.global_config["even_sample_vehicle_class"]
        self.need_default_vehicle = self.engine.global_config["default_vehicle_in_traffic"]
        self.is_ego_vehicle_replay = self.engine.global_config["agent_policy"] == ReplayEgoCarPolicy
        self._filter_overlapping_car = self.engine.global_config["filter_overlapping_car"]

    def before_step(self, *args, **kwargs):
        self._obj_to_clean_this_frame = []
        for v in self.spawned_objects.values():
            if self.engine.has_policy(v.id, ScenarioIDMPolicy):
                p = self.engine.get_policy(v.name)
                if p.arrive_destination:
                    self._obj_to_clean_this_frame.append(self._obj_id_to_scenario_id[v.id])
                else:
                    do_speed_control = self.episode_step % self.IDM_ACT_BATCH_SIZE == p.policy_index
                    v.before_step(p.act(do_speed_control))

    def before_reset(self):
        super(ScenarioTrafficManager, self).before_reset()
        self._obj_to_clean_this_frame = []
        reset_vehicle_type_count(self.np_random)

    def after_reset(self):
        self._scenario_id_to_obj_id = {}
        self._obj_id_to_scenario_id = {}
        self._static_car_id = set()
        self._moving_car_id = set()
        self._noise_object_id = set()
        self._non_noise_object_id = set()
        self.idm_policy_count = 0
        for scenario_id, track in self.current_traffic_data.items():
            if track["type"] == MetaDriveType.VEHICLE:
                self.spawn_vehicle(scenario_id, track)
            elif track["type"] == MetaDriveType.CYCLIST:
                self.spawn_cyclist(scenario_id, track)
            elif track["type"] == MetaDriveType.PEDESTRIAN:
                self.spawn_pedestrian(scenario_id, track)
            elif track["type"] in [MetaDriveType.TRAFFIC_CONE, MetaDriveType.TRAFFIC_BARRIER]:
                cls = TrafficBarrier if track["type"] == MetaDriveType.TRAFFIC_BARRIER else TrafficCone
                self.spawn_static_object(cls, scenario_id, track)
            else:
                logger.warning("Do not support {}".format(track["type"]))

    def after_step(self, *args, **kwargs):
        if self.episode_step < self.current_scenario_length:
            replay_done = False
            for scenario_id, track in self.current_traffic_data.items():
                if scenario_id not in self._scenario_id_to_obj_id:
                    if track["type"] == MetaDriveType.VEHICLE:
                        self.spawn_vehicle(scenario_id, track)
                    elif track["type"] == MetaDriveType.CYCLIST:
                        self.spawn_cyclist(scenario_id, track)
                    elif track["type"] == MetaDriveType.PEDESTRIAN:
                        self.spawn_pedestrian(scenario_id, track)
                    elif track["type"] in [MetaDriveType.TRAFFIC_CONE, MetaDriveType.TRAFFIC_BARRIER]:
                        cls = TrafficBarrier if track["type"] == MetaDriveType.TRAFFIC_BARRIER else TrafficCone
                        self.spawn_static_object(cls, scenario_id, track)
                    else:
                        logger.info("Do not support {}".format(track["type"]))
                elif self.has_policy(self._scenario_id_to_obj_id[scenario_id], ReplayTrafficParticipantPolicy):
                    # static object will not be cleaned!
                    policy = self.get_policy(self._scenario_id_to_obj_id[scenario_id])
                    if policy.is_current_step_valid:
                        policy.act()
                    else:
                        self._obj_to_clean_this_frame.append(scenario_id)
        else:
            replay_done = True
            # clean replay vehicle
            for scenario_id, obj_id in self._scenario_id_to_obj_id.items():
                if self.has_policy(obj_id, ReplayTrafficParticipantPolicy) and not self.is_static_object(obj_id):
                    self._obj_to_clean_this_frame.append(scenario_id)

        for scenario_id in list(self._obj_to_clean_this_frame):
            obj_id = self._scenario_id_to_obj_id.pop(scenario_id)
            _scenario_id = self._obj_id_to_scenario_id.pop(obj_id)
            assert _scenario_id == scenario_id
            self.clear_objects([obj_id])

        return dict(default_agent=dict(replay_done=replay_done))

    @property
    def current_traffic_data(self):
        return self.engine.data_manager.current_scenario["tracks"]

    @property
    def sdc_track_index(self):
        return str(self.engine.data_manager.current_scenario[SD.METADATA][SD.SDC_ID])

    @property
    def sdc_scenario_id(self):
        return self.sdc_track_index

    @property
    def sdc_object_id(self):
        return self.engine.agent_manager.agent_to_object(DEFAULT_AGENT)

    @property
    def current_scenario_length(self):
        return self.engine.data_manager.current_scenario_length

    def spawn_vehicle(self, v_id, track):
        state = parse_object_state(track, self.episode_step)

        # for each vehicle, we would like to know if it is static
        if v_id not in self._static_car_id and v_id not in self._moving_car_id:
            valid_points = track["state"]["position"][np.where(track["state"]["valid"])]
            moving = np.max(np.std(valid_points, axis=0)[:2]) > self.STATIC_THRESHOLD
            set_to_add = self._moving_car_id if moving else self._static_car_id
            set_to_add.add(v_id)

        # don't create in these two conditions
        if not state["valid"] or (self.engine.global_config["no_static_vehicles"] and v_id in self._static_car_id):
            return

        # if collision don't generate, unless ego car is in replay mode
        ego_pos = self.ego_vehicle.position
        heading_dist, side_dist = self.ego_vehicle.convert_to_local_coordinates(state["position"], ego_pos)
        if not self.is_ego_vehicle_replay and self._filter_overlapping_car and \
                abs(heading_dist) < self.GENERATION_FORWARD_CONSTRAINT and \
                abs(side_dist) < self.GENERATION_SIDE_CONSTRAINT:
            return

        # create vehicle
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
            vehicle_class = get_vehicle_type(
                float(state["length"]), None if self.even_sample_v else self.np_random, self.need_default_vehicle
            )
        obj_name = v_id if self.engine.global_config["force_reuse_object_name"] else None
        v = self.spawn_object(
            vehicle_class, position=state["position"], heading=state["heading"], vehicle_config=v_config, name=obj_name
        )
        self._scenario_id_to_obj_id[v_id] = v.name
        self._obj_id_to_scenario_id[v.name] = v_id

        # add policy
        start_index, end_index = get_max_valid_indicis(track, self.episode_step)  # real end_index is end_index-1
        moving = track["state"]["position"][start_index][..., :2] - track["state"]["position"][end_index - 1][..., :2]
        length_ok = np.linalg.norm(moving) > self.IDM_CREATE_MIN_LENGTH
        heading_ok = abs(wrap_to_pi(self.ego_vehicle.heading_theta - state["heading"])) < np.pi / 2
        idm_ok = heading_dist < self.IDM_CREATE_FORWARD_CONSTRAINT and abs(
            side_dist
        ) < self.IDM_CREATE_SIDE_CONSTRAINT and heading_ok
        need_reactive_traffic = self.engine.global_config["reactive_traffic"]
        if not need_reactive_traffic or v_id in self._static_car_id or not idm_ok or not length_ok:
            policy = self.add_policy(v.name, ReplayTrafficParticipantPolicy, v, track)
            policy.act()
        else:
            idm_route = get_idm_route(track["state"]["position"][start_index:end_index][..., :2])
            # only not static and behind ego car, it can get reactive policy
            self.add_policy(
                v.name, ScenarioIDMPolicy, v, self.generate_seed(), idm_route,
                self.idm_policy_count % self.IDM_ACT_BATCH_SIZE
            )
            # no act() is required for IDMPolicy
            self.idm_policy_count += 1

    def spawn_pedestrian(self, scenario_id, track):
        state = parse_object_state(track, self.episode_step)
        if not state["valid"]:
            return
        obj_name = scenario_id if self.engine.global_config["force_reuse_object_name"] else None
        obj = self.spawn_object(
            Pedestrian,
            name=obj_name,
            position=state["position"],
            heading_theta=state["heading"],
        )
        self._scenario_id_to_obj_id[scenario_id] = obj.name
        self._obj_id_to_scenario_id[obj.name] = scenario_id
        policy = self.add_policy(obj.name, ReplayTrafficParticipantPolicy, obj, track)
        policy.act()

    def spawn_cyclist(self, scenario_id, track):
        state = parse_object_state(track, self.episode_step)
        if not state["valid"]:
            return
        obj_name = scenario_id if self.engine.global_config["force_reuse_object_name"] else None
        obj = self.spawn_object(
            Cyclist,
            name=obj_name,
            position=state["position"],
            heading_theta=state["heading"],
        )
        self._scenario_id_to_obj_id[scenario_id] = obj.name
        self._obj_id_to_scenario_id[obj.name] = scenario_id
        policy = self.add_policy(obj.name, ReplayTrafficParticipantPolicy, obj, track)
        policy.act()

    def spawn_static_object(self, cls, scenario_id, track):
        # some
        if scenario_id not in self._noise_object_id and scenario_id not in self._non_noise_object_id:
            valid_length = np.sum(track["state"]["valid"])
            set_to_add = self._noise_object_id if valid_length < self.MIN_VALID_FRAME_LEN else self._non_noise_object_id
            set_to_add.add(scenario_id)
        if scenario_id in self._noise_object_id:
            return

        state = parse_object_state(track, self.episode_step)
        if not state["valid"]:
            return
        obj_name = scenario_id if self.engine.global_config["force_reuse_object_name"] else None
        obj = self.spawn_object(
            cls,
            name=obj_name,
            position=state["position"],
            heading_theta=state["heading"],
        )
        self._scenario_id_to_obj_id[scenario_id] = obj.name
        self._obj_id_to_scenario_id[obj.name] = scenario_id

    def get_state(self):
        # Record mapping from original_id to new_id
        ret = {}
        ret[SD.ORIGINAL_ID_TO_OBJ_ID] = copy.deepcopy(self._scenario_id_to_obj_id)
        ret[SD.OBJ_ID_TO_ORIGINAL_ID] = copy.deepcopy(self._obj_id_to_scenario_id)
        return ret

    @property
    def ego_vehicle(self):
        return self.engine.agents[DEFAULT_AGENT]

    def is_static_object(self, obj_id):
        return isinstance(self.spawned_objects[obj_id], TrafficBarrier) \
               or isinstance(self.spawned_objects[obj_id], TrafficCone)

    @property
    def obj_id_to_scenario_id(self):
        # For outside access, we return traffic vehicles and ego car
        ret = copy.copy(self._obj_id_to_scenario_id)
        ret[self.sdc_object_id] = self.sdc_scenario_id
        return ret

    @property
    def scenario_id_to_obj_id(self):
        # For outside access, we return traffic vehicles and ego car
        ret = copy.copy(self._scenario_id_to_obj_id)
        ret[self.sdc_scenario_id] = self.sdc_object_id
        return ret
