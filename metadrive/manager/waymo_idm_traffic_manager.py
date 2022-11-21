import copy
from collections import namedtuple, OrderedDict

import numpy as np

from metadrive.component.lane.waymo_lane import WayPointLane
from metadrive.component.vehicle.vehicle_type import SVehicle
from metadrive.manager.waymo_traffic_manager import WaymoTrafficManager
from metadrive.policy.idm_policy import WaymoIDMPolicy
from metadrive.utils.waymo_utils.waymo_utils import AgentType

static_vehicle_info = namedtuple("static_vehicle_info", "position heading")


def handler(signum, frame):
    raise Exception("end of time")


class WaymoIDMTrafficManager(WaymoTrafficManager):
    TRAJ_WIDTH = 1.2
    DEST_REGION = 5
    MIN_DURATION = 20
    ACT_FREQ = 5
    MAX_HORIZON = 100

    def __init__(self):
        super(WaymoIDMTrafficManager, self).__init__()
        self.seed_trajs = {}
        self.v_id_to_destination = OrderedDict()
        self.v_id_to_stop_time = OrderedDict()

    def before_reset(self):
        super(WaymoIDMTrafficManager, self).before_reset()

    def after_reset(self):
        # try:
        self.v_id_to_destination = OrderedDict()
        self.v_id_to_stop_time = OrderedDict()
        self.count = 0
        if self.engine.global_random_seed not in self.seed_trajs:
            traffic_traj_data = {}
            for v_id, type_traj in self.current_traffic_data.items():
                if type_traj["type"] == AgentType.VEHICLE and v_id != self.sdc_index:
                    init_info = self.parse_vehicle_state(
                        type_traj["state"], self.engine.global_config["traj_start_index"]
                    )
                    dest_info = self.parse_vehicle_state(
                        type_traj["state"], self.engine.global_config["traj_end_index"], check_last_state=True
                    )
                    if not init_info["valid"]:
                        continue
                    if np.linalg.norm(np.array(init_info["position"]) - np.array(dest_info["position"])) < 1:
                        full_traj = static_vehicle_info(init_info["position"], init_info["heading"])
                        static = True
                    else:
                        full_traj = self.parse_full_trajectory(type_traj["state"])
                        if len(full_traj) < self.MIN_DURATION:
                            full_traj = static_vehicle_info(init_info["position"], init_info["heading"])
                            static = True
                        else:
                            full_traj = WayPointLane(full_traj, width=self.TRAJ_WIDTH)
                            static = False
                    traffic_traj_data[v_id] = {
                        "traj": full_traj,
                        "init_info": init_info,
                        "static": static,
                        "dest_info": dest_info,
                        "is_sdc": False
                    }

                elif type_traj["type"] == AgentType.VEHICLE and v_id == self.sdc_index:
                    # set Ego V velocity
                    init_info = self.parse_vehicle_state(
                        type_traj["state"], self.engine.global_config["traj_start_index"]
                    )
                    traffic_traj_data["sdc"] = {
                        "traj": None,
                        "init_info": init_info,
                        "static": False,
                        "dest_info": None,
                        "is_sdc": True
                    }
            self.seed_trajs[self.engine.global_random_seed] = traffic_traj_data
        policy_count = 0
        for v_traj_id, data in self.current_traffic_traj.items():
            if data["static"] and self.engine.global_config["no_static_traffic_vehicle"]:
                continue
            if v_traj_id == "sdc":
                init_info = data["init_info"]
                ego_v = list(self.engine.agent_manager.active_agents.values())[0]
                ego_v.set_velocity(init_info["velocity"])
                ego_v.set_heading_theta(init_info["heading"], rad_to_degree=False)
                ego_v.set_position(init_info["position"])
                continue
            init_info = data["init_info"]
            v_config = copy.deepcopy(self.engine.global_config["vehicle_config"])
            v_config.update(
                dict(
                    show_navi_mark=False,
                    show_dest_mark=False,
                    enable_reverse=False,
                    show_lidar=False,
                    show_lane_line_detector=False,
                    show_side_detector=False,
                    need_navigation=False,
                )
            )
            v = self.spawn_object(
                SVehicle, position=init_info["position"], heading=init_info["heading"], vehicle_config=v_config
            )
            self.v_id_to_destination[v.id] = np.array(data["dest_info"]["position"])
            self.v_id_to_stop_time[v.id] = 0
            if data["static"]:
                # static vehicle
                v.set_position(v.position, height=0.8)
                v.set_velocity((0, 0))
                v.set_static(True)
            else:
                v.set_position(v.position, height=0.8)
                self.add_policy(
                    v.id, WaymoIDMPolicy, v, self.generate_seed(), data["traj"], policy_count % self.ACT_FREQ
                )
                v.set_velocity(init_info['velocity'])
                policy_count += 1

    def before_step(self, *args, **kwargs):
        for v in self.spawned_objects.values():
            if self.engine.has_policy(v.id):
                p = self.engine.get_policy(v.name)
                do_speed_control = (p.policy_index + self.count) % self.ACT_FREQ == 0
                v.before_step(p.act(do_speed_control))

    def after_step(self, *args, **kwargs):
        self.count += 1
        vehicles_to_clear = []
        # LQY: modify termination condition
        for v in self.spawned_objects.values():
            if not self.engine.has_policy(v.name):
                continue
            if v.speed < 1:
                self.v_id_to_stop_time[v.id] += 1
            else:
                self.v_id_to_stop_time[v.id] = 0
            dist_to_dest = np.linalg.norm(v.position - self.v_id_to_destination[v.id])
            if dist_to_dest < self.DEST_REGION or self.v_id_to_stop_time[v.id] > self.MAX_HORIZON:
                vehicles_to_clear.append(v.id)
        self.clear_objects(vehicles_to_clear)

    @property
    def current_traffic_traj(self):
        return self.seed_trajs[self.engine.global_random_seed]
