import copy
import os
import os.path as osp
import shutil

import argparse
import copy
import logging
import os
import os.path as osp
import shutil
from collections import namedtuple

import cv2
import numpy as np
import seaborn as sns
from metadrive.component.lane.point_lane import PointLane
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.vehicle.vehicle_type import VaryingShapeVehicle
from metadrive.component.vehicle_navigation_module.trajectory_navigation import WaymoTrajectoryNavigation
from metadrive.constants import RENDER_MODE_ONSCREEN, CamMask
from metadrive.constants import TerminationState, DEFAULT_AGENT
from metadrive.engine import get_engine
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.base_env import BaseEnv
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.manager.agent_manager import AgentManager
from metadrive.manager.base_manager import BaseManager
from metadrive.manager.waymo_data_manager import WaymoDataManager
from metadrive.manager.waymo_map_manager import WaymoMapManager
from metadrive.manager.waymo_traffic_manager import WaymoTrafficManager
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.utils import get_np_random
from metadrive.utils.coordinates_shift import waymo_2_metadrive_heading
from metadrive.utils.coordinates_shift import waymo_2_metadrive_position
from metadrive.utils.error_class import NavigationError
from metadrive.utils.math_utils import clip
from metadrive.utils.space import ParameterSpace, ConstantSpace, BoxSpace
from metadrive.utils.waymo_utils.waymo_utils import AgentType
from panda3d.core import TransparencyAttrib, LineSegs, NodePath
from tqdm.auto import tqdm
import numpy as np
from metadrive.utils.coordinates_shift import waymo_2_metadrive_position
from metadrive.utils.math_utils import clip
from tqdm import tqdm

from newcopo.metadrive_scenario.marl_envs.marl_single_waymo_env import MARLSingleWaymoEnv, \
    MARL_SINGLE_WAYMO_ENV_CONFIG, image_to_video

WAYMO_DATASET_PATH = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), "dataset", "env_num_1165_waymo")

MARL_WAYMO_ENV_CONFIG = copy.deepcopy(MARL_SINGLE_WAYMO_ENV_CONFIG)
MARL_WAYMO_ENV_CONFIG.update(
    {
        # ===== Set scene to [0, 1000] =====
        "start_case_index": 0,
        "case_num": 1000,
        "dataset_path": WAYMO_DATASET_PATH,

        "distance_penalty": 0
    }
)


STANDARD_WAYMO_ENV_PATH = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))))

STATIC_VEHICLE_NO_ROUTE = "static_vehicle_no_route"


def process_expert_trajectory(track_data):
    """
    track_data: A np array with shape [198, 10]
    """
    # if track_data.shape[0] > 198:
    #     # Some time the trajectory might contain full 200 frames.
    #     track_data = track_data[:198]
    # if track_data.shape[0] < 198:
    #     track_data = np.concatenate([track_data, np.zeros([198 - track_data.shape[0], track_data.shape[1]])], axis=0)

    track_data = track_data[:20]

    # POSITION
    # First 2 dim is position.
    pos = track_data[:, :2]  # [198, 2]

    invalid_mask = pos.mean(axis=1) == 0  # [198, ] True if this frame is missing
    new_pos = pos - pos[0]  # relative position from the init position.

    # A little scale
    new_pos = new_pos / 50

    # Filter out missing positions:
    new_pos[invalid_mask] = 0

    # VELOCITY
    velocity = track_data[:, 7: 7 + 2]
    velocity = velocity / 30

    # LENGTH and WIDTH
    length = track_data[:, 3] / 10
    width = track_data[:, 4] / 10

    # HEADING
    heading = track_data[:, 6] / (np.pi * 2)

    valid = track_data[:, 9]

    ret = np.concatenate([new_pos, velocity, np.stack([heading, length, width, valid], axis=-1)], axis=-1)
    # ret is in [198, 8]

    return ret


class WaymoVehicle(VaryingShapeVehicle):
    PARAMETER_SPACE = ParameterSpace(dict(
        max_speed=ConstantSpace(80),

        # Change dynamics model!
        wheel_friction=ConstantSpace(0.9),
        max_engine_force=BoxSpace(750, 850),
        max_brake_force=BoxSpace(80, 180),
        max_steering=ConstantSpace(40),

        # Dynamics 1
        # wheel_friction=ConstantSpace(1.2),
        # max_engine_force=ConstantSpace(1200),
        # max_brake_force=ConstantSpace(300),
        # max_steering=ConstantSpace(80),

        # Dynamics 2
        # wheel_friction=ConstantSpace(1.8),
        # max_engine_force=ConstantSpace(1800),
        # max_brake_force=ConstantSpace(400),
        # max_steering=ConstantSpace(80),

        # Dynamics 3
        # wheel_friction=ConstantSpace(2.4),
        # max_engine_force=ConstantSpace(10000),
        # max_brake_force=ConstantSpace(600),
        # max_steering=ConstantSpace(60),
    ))


class MAWaymoTrajectoryNavigation(WaymoTrajectoryNavigation):
    def __init__(
            self,
            show_navi_mark: bool = False,
            random_navi_mark_color=False,
            show_dest_mark=False,
            show_line_to_dest=False,
            panda_color=None,
            name=None,
            vehicle_config=None
    ):
        assert vehicle_config is not None

        """
        This class define a helper for localizing vehicles and retrieving navigation information.
        It now only support from first block start to the end node, but can be extended easily.
        """
        # self.engine = engine
        self.name = name

        # Make sure these variables are filled when making new subclass
        # self.map = None
        # self.checkpoints = None
        # self.current_ref_lanes = None
        # self.next_ref_lanes = None
        # self.final_lane = None
        # self.current_lane = None
        self.vehicle_config = vehicle_config

        self._target_checkpoints_index = None
        self._navi_info = np.zeros((self.navigation_info_dim,), dtype=np.float32)  # navi information res

        # Vis
        self._show_navi_info = (
                self.engine.mode == RENDER_MODE_ONSCREEN and not self.engine.global_config["debug_physics_world"])
        self.origin = NodePath("navigation_sign") if self._show_navi_info else None
        self.navi_mark_color = (0.6, 0.8, 0.5) if not random_navi_mark_color else get_np_random().rand(3)
        if panda_color is not None:
            assert len(panda_color) == 3 and 0 <= panda_color[0] <= 1
            self.navi_mark_color = tuple(panda_color)
        self.navi_arrow_dir = [0, 0]
        self._dest_node_path = None
        self._goal_node_path = None

        self._node_path_list = []

        self._line_to_dest = None
        self._line_to_navi = None
        self._show_line_to_dest = show_line_to_dest
        if self._show_navi_info:
            # nodepath
            self._line_to_dest = self.origin.attachNewNode("line")
            self._goal_node_path = self.origin.attachNewNode("target")
            self._dest_node_path = self.origin.attachNewNode("dest")

            self._node_path_list.append(self._line_to_dest)
            self._node_path_list.append(self._goal_node_path)
            self._node_path_list.append(self._dest_node_path)

            if show_navi_mark:
                navi_point_model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
                navi_point_model.reparentTo(self._goal_node_path)
            if show_dest_mark:
                dest_point_model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
                dest_point_model.reparentTo(self._dest_node_path)
            if show_line_to_dest:
                line_seg = LineSegs("line_to_dest")
                line_seg.setColor(self.navi_mark_color[0], self.navi_mark_color[1], self.navi_mark_color[2], 1.0)
                line_seg.setThickness(4)
                self._dynamic_line_np = NodePath(line_seg.create(True))

                self._node_path_list.append(self._dynamic_line_np)

                self._dynamic_line_np.reparentTo(self.origin)
                self._line_to_dest = line_seg

            show_line_to_navi_mark = self.vehicle_config["show_line_to_navi_mark"]
            self._show_line_to_navi_mark = show_line_to_navi_mark
            if show_line_to_navi_mark:
                line_seg = LineSegs("line_to_dest")
                line_seg.setColor(self.navi_mark_color[0], self.navi_mark_color[1], self.navi_mark_color[2], 1.0)
                line_seg.setThickness(4)
                self._dynamic_line_np_2 = NodePath(line_seg.create(True))

                self._node_path_list.append(self._dynamic_line_np_2)

                self._dynamic_line_np_2.reparentTo(self.origin)
                self._line_to_navi = line_seg

            self._goal_node_path.setTransparency(TransparencyAttrib.M_alpha)
            self._dest_node_path.setTransparency(TransparencyAttrib.M_alpha)

            self._goal_node_path.setColor(
                self.navi_mark_color[0], self.navi_mark_color[1], self.navi_mark_color[2], 0.7
            )
            self._dest_node_path.setColor(
                self.navi_mark_color[0], self.navi_mark_color[1], self.navi_mark_color[2], 0.7
            )
            self._goal_node_path.hide(CamMask.AllOn)
            self._dest_node_path.hide(CamMask.AllOn)
            self._goal_node_path.show(CamMask.MainCam)
            self._dest_node_path.show(CamMask.MainCam)
        logging.debug("Load Vehicle Module: {}".format(self.__class__.__name__))

        assert isinstance(self.name, str), self.name

    def reset(self, map=None, current_lane=None, destination=None, random_seed=None):
        # self.map = map
        self.map = None
        if self.reference_trajectory is not None:
            self.set_route(None, None)

    def set_route(self, current_lane_index: str, destination: str):
        self.checkpoints = self.discretize_reference_trajectory()
        self._target_checkpoints_index = [0, 1] if len(self.checkpoints) >= 2 else [0, 0]
        # update routing info
        # assert len(self.checkpoints
        #            ) >= 2, "Can not find a route from {} to {}".format(current_lane_index[0], destination)

        self._navi_info.fill(0.0)
        # self.current_ref_lanes = [self.reference_trajectory]
        # self.next_ref_lanes = None
        # self.current_lane = self.final_lane = self.reference_trajectory

        # self.current_ref_lanes = None
        # self.next_ref_lanes = None
        # self.current_lane = None
        # self.final_lane = None

        if self._dest_node_path is not None:
            check_point = self.reference_trajectory.end
            self._dest_node_path.setPos(check_point[0], -check_point[1], 1.8)

    @property
    def engine(self):
        return get_engine()

    @property
    def current_lane(self):
        return self.reference_trajectory

    @property
    def current_ref_lanes(self):
        return [self.reference_trajectory]

    @property
    def reference_trajectory(self):
        agent_name = self.vehicle_config["agent_name"]
        if agent_name not in self.engine.map_manager.current_routes:
            from metadrive.utils.error_class import NavigationError
            raise NavigationError(
                "{} has no record in MapManager: {}".format(agent_name, self.engine.map_manager.current_routes.keys())
            )
        elif self.engine.map_manager.current_routes[agent_name] == STATIC_VEHICLE_NO_ROUTE:
            return None

        # Use deepcopy to avoid circular reference
        return self.engine.map_manager.current_routes[agent_name]

    def get_trajectory(self):
        """This function breaks Multi-agent Waymo Env since we don't set this in map_manager."""
        raise ValueError()
        # agent_name = self.vehicle_config["agent_name"]
        # if agent_name not in self.engine.map_manager.current_routes:
        #     from metadrive.utils.error_class import NavigationError
        #     raise NavigationError(
        #         "{} has no record in MapManager: {}".format(agent_name, self.engine.map_manager.current_routes.keys())
        #     )
        # elif self.engine.map_manager.current_routes[agent_name] == STATIC_VEHICLE_NO_ROUTE:
        #     return None
        #
        # # Use deepcopy to avoid circular reference
        # return copy.deepcopy(self.engine.map_manager.current_routes[agent_name])

    def destroy(self):
        # self.map = None
        # self.checkpoints = None
        # self.current_ref_lanes = None
        # self.next_ref_lanes = None
        # self.final_lane = None
        # self.current_lane = None
        # self.reference_trajectory = None
        super(WaymoTrajectoryNavigation, self).destroy()

    def before_reset(self):
        self.map = None
        self.checkpoints = None
        # self.current_ref_lanes = None
        self.next_ref_lanes = None
        self.final_lane = None
        # self.current_lane = None
        # self.reference_trajectory = None


MARL_SINGLE_WAYMO_ENV_CONFIG = {
    "dataset_path": None,
    "num_agents": -1,  # Automatically determine how many agents
    "start_case_index": 551,  # ===== Set the scene to 551 =====
    "case_num": 1,
    "waymo_env": True,
    "vehicle_config": {
        "agent_name": None,
        "spawn_lane_index": None,
        "navigation_module": MAWaymoTrajectoryNavigation
    },
    "no_static_traffic_vehicle": False,  # Allow static vehicles!
    "horizon": 200,  # The environment will end when environmental steps reach 200
    "delay_done": 0,  # If agent dies, it will be removed immediately from the scene.
    # "sequential_seed": False,
    # "save_memory": False,
    # "save_memory_max_len": 50,

    "store_map": True,
    "store_map_buffer_size": 10,

    # ===== New config ===
    "randomized_dynamics": None,  #####

    "discrete_action_dim": 5,

    "relax_out_of_road_done": False,

    "replay_traffic_vehicle": False

}

class MAWaymoMapManager(WaymoMapManager):
    """
    Compared to WaymoMapManager, we force to load maps so that we can determine
    the possible spawn positions of vehicles
    """

    def __init__(self):

        WaymoMapManager.__init__(self)

        # PZH: These two dicts are newly introduced. Their keys are the agent name of the vehicle, e.g. "111" or "sdc".
        self.current_routes = dict()
        self.dest_points = dict()

    def update_route(self):
        """
        This function computes the routes for all vehicles.
        """

        _debug_memory = False
        if _debug_memory:
            # inner psutil function
            def process_memory():
                import psutil
                import os
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                return mem_info.rss

            cm = process_memory()

        data = self.engine.data_manager.get_case(self.engine.global_random_seed, should_copy=False)

        if _debug_memory:
            lm = process_memory()
            if lm - cm != 0:
                print("[Update Route] {}:  Reset! Mem Change {:.3f}MB".format(0, (lm - cm) / 1e6))
            cm = lm

        self.current_routes = {}
        self.dest_points = {}

        if _debug_memory:
            lm = process_memory()
            if lm - cm != 0:
                print("[Update Route] {}:  Reset! Mem Change {:.3f}MB".format(1, (lm - cm) / 1e6))
            cm = lm

        for v_id, track in data["tracks"].items():
            if track["type"] != AgentType.VEHICLE:
                continue

            if v_id == data["sdc_index"]:
                v_id = "sdc"

            traj = WaymoTrafficManager.parse_full_trajectory(track["state"])

            # traj_dis = sum([np.linalg.norm(d) for d in traj[:-1] - traj[1:]])
            # traj_count = len(traj)

            if len(traj) < 2:
                # Something wrong in the data, remove this car.
                continue

            if np.linalg.norm(traj[0] - traj[-1]) < 10:
                # This car is not moving, we should discard it.
                # print("The track for vehicle {} is too short: {}. Skip!".format(
                #     v_id, np.linalg.norm(traj[0] - traj[-1])))
                self.current_routes[v_id] = STATIC_VEHICLE_NO_ROUTE
                continue

            self.current_routes[v_id] = PointLane(traj, width=1.5)

            last_state = WaymoTrafficManager.parse_vehicle_state(
                track["state"],
                time_idx=-1,  # Take the final state
                check_last_state=True
            )
            last_position = last_state["position"]
            self.dest_points[v_id] = last_position

        if _debug_memory:
            lm = process_memory()
            if lm - cm != 0:
                print("[Update Route] {}:  Reset! Mem Change {:.3f}MB".format(2, (lm - cm) / 1e6))
            cm = lm

    def destroy(self):
        self.maps = None
        self.current_map = None
        BaseManager.destroy(self)

    def unload_map(self, map):
        map.detach_from_world()
        map.destroy()
        del map
        self.current_map = None

    def before_reset(self):
        # remove map from world before adding
        if self.current_map is not None:
            self.unload_map(self.current_map)



        self.current_routes = {}
        self.dest_points = {}

        self.sdc_start = None
        self.sdc_destinations = []
        self.sdc_dest_point = None
        self.current_sdc_route = None
        self.current_map = None
        self.maps = None

    def reset(self):

        self.current_routes = {}
        self.dest_points = {}

        return super(MAWaymoMapManager, self).reset()


# PZH: from WaymoIDMTrafficManager
# static_vehicle_info = namedtuple("static_vehicle_info", "position heading")


class MAWaymoAgentManager(AgentManager):
    # PZH: from WaymoIDMTrafficManager
    TRAJ_WIDTH = 1.2
    DEST_REGION = 5
    MIN_DURATION = 20
    ACT_FREQ = 5

    # MAX_HORIZON = 100

    def __init__(self, init_observations, init_action_space, store_map, store_map_buffer_size):
        AgentManager.__init__(self, init_observations=init_observations, init_action_space=init_action_space)

        # self.current_traffic_data = None
        # self.count = 0
        # self.sdc_index = None
        # self.vid_to_obj = None

        # PZH: from WaymoIDMTrafficManager
        # self.seed_trajs = DataBuffer(store_data_buffer_size=store_map_buffer_size if store_map else None)
        # self.seed_trajs = {}
        # self.seed_trajs_index = deque(maxlen=max_len)
        # self.save_memory = save_memory
        # self.max_len = max_len

        self.dynamics_parameters_mean = None
        self.dynamics_parameters_std = None
        self.dynamics_function = None
        self.latent_dict = None

    def _get_vehicle_type(self):
        return WaymoVehicle

    def set_dynamics_parameters_distribution(self, dynamics_parameters_mean=None, dynamics_parameters_std=None, dynamics_function=None, latent_dict=None):
        # dynamics_parameters_mean in [-1, 1]
        self.dynamics_parameters_mean = dynamics_parameters_mean
        self.dynamics_parameters_std = dynamics_parameters_std
        self.dynamics_function = dynamics_function
        if latent_dict is not None:
            self.latent_dict = latent_dict

        if self.engine.global_config["randomized_dynamics"] in ["nn", "nnstd", "dynamics_policy"]:
            self._dynamics_parameters_pointer = 0

        assert self.engine.global_config["randomized_dynamics"] in [None, "nn", "nnstd", "mean", "std", "gmm", "naive", "dynamics_policy"]

    def get_expert_trajectory(self, v_traj_id):
        # assert "latent_trajectory_len" in self.engine.global_config
        # assert self.engine.global_config["latent_trajectory_len"] > 0
        expert_id = v_traj_id  # Create a new variable to avoid affect "v_traj_id"
        if expert_id == "sdc":
            expert_id = self.current_traffic_data["sdc_index"]
        this_track = self.current_traffic_data["tracks"][expert_id]['state']
        expert_traj = process_expert_trajectory(this_track)
        return expert_traj

    def reset(self):
        """
        We should reconfigure all spawn locations according to the latest map

        """

        # Clear existing data:
        self._agent_to_object = {}
        self._object_to_agent = {}
        self._active_objects = {}
        self._dying_objects = {}
        self._agents_finished_this_frame = dict()
        self.observations = dict()
        self.observation_spaces = dict()
        self.action_spaces = dict()

        # PZH: Copy from WaymoIDMTrafficManager
        episode_created_agents = {}
        # self.v_id_to_destination = OrderedDict()
        # self.v_id_to_stop_time = OrderedDict()
        # self.count = 0

        _debug_memory_usage = False

        if _debug_memory_usage:
            def process_memory():
                import psutil
                import os
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                return mem_info.rss

            cm = process_memory()

        # seed = self.engine.global_random_seed
        # print('11111')

        if True:
            # if seed not in self.seed_trajs:

            # Load the trajectories for this particular seed
            traffic_traj_data = {}
            for v_id, type_traj in self.current_traffic_data["tracks"].items():
                if type_traj["type"] == AgentType.VEHICLE and v_id != self.current_traffic_data["sdc_index"]:
                    init_info = self.parse_vehicle_state(
                        type_traj["state"], self.engine.global_config["traj_start_index"]
                    )
                    dest_info = self.parse_vehicle_state(
                        type_traj["state"], self.engine.global_config["traj_end_index"], check_last_state=True
                    )
                    if not init_info["valid"]:
                        continue
                    if np.linalg.norm(np.array(init_info["position"]) - np.array(dest_info["position"])) < 1:
                        # full_traj = static_vehicle_info(init_info["position"], init_info["heading"])
                        static = True
                    else:
                        full_traj = self.parse_full_trajectory(type_traj["state"])
                        if len(full_traj) < self.MIN_DURATION:
                            # full_traj = static_vehicle_info(init_info["position"], init_info["heading"])
                            static = True
                        else:
                            # full_traj = WayPointLane(full_traj, width=self.TRAJ_WIDTH)
                            static = False
                    traffic_traj_data[v_id] = {
                        # "traj": full_traj,
                        "init_info": init_info,
                        "static": static,
                        "dest_info": dest_info,
                        "is_sdc": False
                    }

                elif type_traj["type"] == AgentType.VEHICLE and v_id == self.current_traffic_data["sdc_index"]:
                    # set Ego V velocity
                    init_info = self.parse_vehicle_state(
                        type_traj["state"], self.engine.global_config["traj_start_index"]
                    )
                    traffic_traj_data["sdc"] = {
                        # "traj": None,
                        "init_info": init_info,
                        "static": False,
                        "dest_info": None,
                        "is_sdc": True
                    }
            # self.seed_trajs[seed] = traffic_traj_data
            # self.seed_trajs_index.append(seed)

        if _debug_memory_usage:
            lm = process_memory()
            if lm - cm != 0:
                print("{}:  Reset! Mem Change {:.3f}MB".format("agent manager 1", (lm - cm) / 1e6))
            cm = lm

        # policy_count = 0
        # for v_count, (v_traj_id, data) in enumerate(self.seed_trajs[seed].items()):
        for v_count, (v_traj_id, data) in enumerate(traffic_traj_data.items()):

            if data["static"] and self.engine.global_config["no_static_traffic_vehicle"]:
                continue

            # assert not data["static"]

            v_traj_id = str(v_traj_id)

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

                    # PZH: Of course we need this
                    # need_navigation=False,
                    need_navigation=True,

                    # === Dynamics ===
                    # max_engine_force=None,
                    # max_brake_force=None,
                    # wheel_friction=None,
                    # max_steering=None
                )
            )

            # Allow changing the shape of vehicle according to the data
            width = length = height = None
            if data["init_info"] is not None and data["dest_info"] is not None:
                width = (data["init_info"]["width"] + data["dest_info"]["width"]) / 2
                length = (data["init_info"]["length"] + data["dest_info"]["length"]) / 2
            elif data["init_info"] is not None:
                width = data["init_info"]["width"]
                length = data["init_info"]["length"]
            elif data["dest_info"] is not None:
                width = data["dest_info"]["width"]
                length = data["dest_info"]["length"]
            v_config["width"] = width
            v_config["length"] = length

            if data["static"] or self.engine.map_manager.current_routes[v_traj_id] == STATIC_VEHICLE_NO_ROUTE:
                v_config["need_navigation"] = False

            v_config["agent_name"] = v_traj_id

            dynamics_dict = None
            if self.engine.global_config["randomized_dynamics"] and self.dynamics_parameters_mean is not None:
                # Important Note: the sample values should fall into [-1, 1]!

                if self.engine.global_config["randomized_dynamics"] in ["mean", "std"]:
                    assert len(self.dynamics_parameters_mean) == len(self.dynamics_parameters_std)
                    val = np.random.normal(loc=self.dynamics_parameters_mean, scale=self.dynamics_parameters_std)
                elif self.engine.global_config["randomized_dynamics"] in ["nn", "nnstd"]:
                    val = self.dynamics_parameters_mean[self._dynamics_parameters_pointer]
                    self._dynamics_parameters_pointer += 1
                    if self._dynamics_parameters_pointer >= len(self.dynamics_parameters_mean):
                        self._dynamics_parameters_pointer = 0
                else:
                    raise ValueError()

                dynamics_dict = embedding_to_dynamics_parameters(val)
                v_config.update(dynamics_dict)

            if self.engine.global_config["randomized_dynamics"] and self.dynamics_function is not None:
                assert self.dynamics_parameters_mean is None

                # Prepare expert trajectory here
                # expert_traj = None
                # if "latent_trajectory_len" in self.engine.global_config and \
                #     self.engine.global_config["latent_trajectory_len"] > 0:
                #     expert_traj = self.get_expert_trajectory(v_traj_id)

                val, info = self.dynamics_function(
                    environment_seed=self.engine.global_seed,
                    agent_name=v_traj_id,
                    latent_dict=self.latent_dict,
                    # expert_traj=expert_traj
                )
                if val is not None:
                    dynamics_dict = embedding_to_dynamics_parameters(val)
                    v_config.update(dynamics_dict)

            try:

                v = self.spawn_object(
                    object_class=self._get_vehicle_type(),
                    # name=v_traj_id,  # PZH: We should not bind the name of object with name of agent!
                    position=init_info["position"],

                    # heading argument is in degree! init_info["heading"] is in degree too!
                    heading=init_info["heading"],
                    vehicle_config=v_config,

                    # It would be great if we don't force spawn. This is because the old vehicle will be stored in the
                    # buffer (engine._dying_objects) and if there are too much of them the memory will leak.
                    # force_spawn=True
                )

                if self.engine.global_config["randomized_dynamics"] == "naive" and self.dynamics_function is not None:
                    v._dynamics_mode = info["mode"]

                # print("Vehicle is reset: {}. dynamics_dict: {}".format(v.get_dynamics_parameters(), dynamics_dict))

                # print("Current vehicle shape: {}, Should be: {}".format(
                #     (
                #         list(v.system.chassis.shapes)[0].half_extents_with_margin[0] * 2,
                #         list(v.system.chassis.shapes)[0].half_extents_with_margin[1] * 2),
                #     (width, length)
                # ))

            except NavigationError:
                print(
                    "Vehicle {} can not find navigation. Maybe it is out of the road. It's init states are: ".
                    format(v_traj_id), init_info
                )
                continue

            # We set the height to 0.8 for no reason. It just a randomly picked number.
            v.set_position(v.position, height=0.8)

            if data["static"] or \
                    (v.navigation is None) or \
                    (hasattr(v.navigation, "current_ref_lanes") and v.navigation.current_ref_lanes is None) or \
                    (hasattr(v.navigation, "current_ref_lanes") and v.navigation.current_ref_lanes[0] is None):
                v.set_velocity((0, 0))
                v.set_static(True)

                self.put_to_static_list(v)

                valid = False
            else:
                # self.add_policy(
                #     v.id, WaymoIDMPolicy, v, self.generate_seed(), data["traj"], policy_count % self.ACT_FREQ
                # )

                if self.engine.global_config["replay_traffic_vehicle"] and v_traj_id != "sdc":
                    from newcopo.metadrive_scenario.marl_envs.replay_policy import ReplayPolicy
                    self.add_policy(
                        v.id,
                        ReplayPolicy,
                        v,
                        self.generate_seed(),  # data["traj"], policy_count % self.ACT_FREQ
                        vehicle_id=v_traj_id
                    )

                else:
                    self.add_policy(
                        v.id,
                        EnvInputPolicy,
                        v,
                        self.generate_seed(),  # data["traj"], policy_count % self.ACT_FREQ
                    )

                v.set_velocity(init_info['velocity'])
                # policy_count += 1

                valid = True

            episode_created_agents[v_traj_id] = v

            # agent_name = self.next_agent_id()
            # next_config = self.engine.global_config["target_vehicle_configs"]["agent0"]
            # vehicle = self._get_vehicles({agent_name: next_config})[agent_name]
            # new_v_name = vehicle.name

            self._agent_to_object[v_traj_id] = v.id
            self._object_to_agent[v.id] = v_traj_id

            if valid:
                # if self.engine.global_config["randomized_dynamics"] and self.dynamics_function is not None:
                #     assert dynamics_dict is not None, "Vehicle {} does not randomize its dynamics".format(v_traj_id)
                self.observations[v.id] = self._init_observations[DEFAULT_AGENT]
                self.observations[v.id].reset(v)
                self.observation_spaces[v.id] = self._init_observation_spaces[DEFAULT_AGENT]
                self.action_spaces[v.id] = self._init_action_spaces[DEFAULT_AGENT]
                self._active_objects[v.id] = v

            self._check()
            # step_info = vehicle.before_step([0, 0])
            # vehicle.set_static(False)
            # return agent_name, vehicle, step_info

        if _debug_memory_usage:
            lm = process_memory()
            if lm - cm != 0:
                print("{}:  Reset! Mem Change {:.3f}MB".format("agent manager 2", (lm - cm) / 1e6))
            cm = lm

        # Update some intermediate flags
        # self.random_spawn_lane_in_single_agent()
        config = self.engine.global_config
        self._debug = config["debug"]
        self._delay_done = config["delay_done"]
        self._infinite_agents = config["num_agents"] == -1
        self._allow_respawn = config["allow_respawn"]

        # self.episode_created_agents = self._get_vehicles(
        #     config_dict=self.engine.global_config["target_vehicle_configs"]
        # )
        self.episode_created_agents = episode_created_agents

        assert self._delay_done == 0

        # try:
        #     print(
        #         "Map {}, Successfully spawn {} vehicles where {} are controllable by RL policies. Average length of trajectory: {:.3f}. Vehicles ids: {}"
        #         .format(
        #             self.engine.global_random_seed, len(episode_created_agents), len(self.active_agents),
        #             np.mean(
        #                 [
        #                     v.navigation.reference_trajectory.length for v in episode_created_agents.values()
        #                     if v.navigation is not None and v.navigation.reference_trajectory is not None
        #                 ]
        #             ),
        #             self.active_agents.keys()
        #         )
        #     )
        # except Exception:
        #     pass

        if _debug_memory_usage:
            lm = process_memory()
            if lm - cm != 0:
                print("{}:  Reset! Mem Change {:.3f}MB".format("agent manager 3", (lm - cm) / 1e6))
            cm = lm

        self._valid_vehicles_after_reset = set(self.active_agents.keys())

    def random_spawn_lane_in_single_agent(self, *args, **kwargs):
        raise NotImplementedError()

    def _get_vehicles(self, *_, **__):
        raise NotImplementedError()

    def get_policy(self, *_, **__):
        raise NotImplementedError()

    # @property
    # def current_traffic_traj(self):
    #     # PZH: from WaymoIDMTrafficManager
    #     return self.seed_trajs[self.engine.global_random_seed]

    def after_reset(self):
        pass

    @staticmethod
    def parse_vehicle_state(states, time_idx, check_last_state=False):
        ret = {}
        if time_idx >= len(states):
            time_idx = -1
        state = states[time_idx]
        if check_last_state:
            for current_idx in range(len(states) - 1):
                p_1 = states[current_idx][:2]
                p_2 = states[current_idx + 1][:2]
                if np.linalg.norm(p_1 - p_2) > 100:
                    state = states[current_idx]
                    break

        # Little fix: If a vehicle disappear for future 10 frames, then we set it to invalid.
        # Instead of using "invalid" flag from Waymo dataset directly:
        # ret["valid"] = state[9]
        if time_idx != -1:
            valid = states[time_idx: time_idx + 10, 9].max()
        else:
            valid = state[9]

        if valid != state[9]:
            # This frame is lost, we should interpolate values:

            search_states = states[max(time_idx - 5, 0): min(time_idx + 5, len(states))]

            # Interpolation
            fail = True
            if len(search_states) != 0:
                search_valid_mask = search_states[:, 9].astype(bool)
                if search_valid_mask.mean() > 0:
                    state = search_states[search_valid_mask].mean(axis=0)
                    fail = False

            if fail:
                valid = False

        ret["valid"] = valid
        ret["position"] = waymo_2_metadrive_position([state[0], state[1]])
        ret["length"] = state[3]
        ret["width"] = state[4]
        ret["heading"] = waymo_2_metadrive_heading(np.rad2deg(state[6]))
        ret["velocity"] = waymo_2_metadrive_position([state[7], state[8]])

        return ret

    def after_step(self, *args, **kwargs):
        step_infos = self.for_each_active_agents(lambda v: v.after_step())
        return step_infos

    def before_reset(self):
        if not self.INITIALIZED:
            BaseManager.__init__(self)
            self.INITIALIZED = True

        for v in self.get_vehicle_list():
            if v.navigation is not None:
                v.navigation.destroy()
                v.navigation = None

        for v in self.dying_agents.values():
            self._remove_vehicle(v)

        BaseManager.before_reset(self)
        # self.episode_created_agents = None
        # self.current_traffic_data = None
        # seed = self.engine.global_random_seed
        # self.current_traffic_data = copy.deepcopy(self.engine.data_manager.get_case(seed))

    @property
    def current_traffic_data(self):
        return self.engine.data_manager.get_case(self.engine.global_random_seed, should_copy=False)

    @staticmethod
    def parse_full_trajectory(states):

        # TODO: Current implementation does not consider the case when the car is start at a few frame later.
        #  That is, in the first few frames, the car is located at [0, 0] but later suddenly has new positions.

        index = len(states)
        for current_idx in range(len(states) - 1):
            p_1 = states[current_idx][:2]
            p_2 = states[current_idx + 1][:2]
            if np.linalg.norm(p_1 - p_2) > 100:
                index = current_idx
                break

        states = states[:index]
        trajectory = copy.deepcopy(states[:, :2])
        # convert to metadrive coordinate
        trajectory *= [1, -1]

        return trajectory

    def put_to_static_list(self, v):
        # This function uses the delay done mechanism in original AgentManager.
        # We set the timeout to be very large values so the car will be kept static at original place.
        vehicle_name = v.name

        if vehicle_name in self._active_objects:
            self._active_objects.pop(vehicle_name)

        v.set_static(True)
        self._dying_objects[vehicle_name] = [v, 1000000]  # (vehicle, seconds before removing the vehicle)

    def put_vehicles_to_logged_states(self, time_step):
        # Copied from Waymo Traffic Manager
        tracks = self.current_traffic_data["tracks"]

        # TODO: FIXME: Problematic! In some cases the data contains vehicles that are not active currently!
        #  This might happen, e.g., when you skip to time_step 199 but the real MetaDrive env is at time_step 0!!
        #  We should probably create new vehicles for this!

        for v_id, v in self.active_agents.items():

            if v_id == "sdc":
                assert self.current_traffic_data["sdc_index"] in tracks
                info = self.parse_vehicle_state(tracks[self.current_traffic_data["sdc_index"]]["state"],
                                                time_idx=time_step)

            else:
                if v_id not in tracks:
                    print("Vehicle {} not in logged data".format(v_id))
                    continue
                info = self.parse_vehicle_state(tracks[v_id]["state"], time_idx=time_step)

            time_end = time_step > self.engine.global_config["traj_end_index"] and self.engine.global_config[
                "traj_end_index"] != -1
            if (not info["valid"] or time_end):
                self.finish(v_id, ignore_delay_done=True)
                continue

            # Say the vehicle's height is 1.2. We can't set the vehicle height to "v.HEIGHT / 2 + 1" since it's 1.6m
            # Far higher than the lidar's 1.2m
            # The reason we originally set vehicle height to "v.HEIGHT / 2 + 1" is because when reset the environment,
            # we "drop" the vehicle from the sky and let them fall to the ground.
            # But this is not the case in log replay.
            # Since the vehicle height is somewhere around 1.6m, we can simply set the height here to be 0.4m
            # (the default argument of "set_position" function)
            v.set_position(info["position"], 0.4)

            v.set_heading_theta(info["heading"], rad_to_degree=False)
            v.set_velocity(info['velocity'])

        # Unknown issue in the engine that we have to do a little simulation so that the "set_position" can become
        # valid, in a sense that the LiDAR can see objects.
        # If we don't call this line, in visualization we can still see the vehicles moving around. But
        # their physics is not synced yet.
        self.engine.physics_world.dynamic_world.doPhysics(0.02, 1, 0.02)

        # print("Vehicle {} is set to new position {} and heading {}".format(
        #     v_id, info["position"], info["heading"]
        # ))
        # self.count += 1
        # except:
        #     raise ValueError("Can not UPDATE traffic for seed: {}".format(self.engine.global_random_seed))




def _get_action(value, max_value):
    """Transform a discrete value: [0, max_value) into a continuous value [-1, 1]"""
    action = value / (max_value - 1)
    action = action * 2 - 1
    return action

MARLSingleWaymoEnv
class MARLSingleWaymoEnv(WaymoEnv, MultiAgentMetaDrive):
    """
    PZH Note:
    We should strictly spawn N agents at N locations, which are the initial positions of all
    "valid" vehicles in one real waymo scene.

    Problems:
    1. How to determine a vehicles in the scene is valid?
    2. How to spawn RL vehicles to each valid vehicle? (Modify TrafficManager, AgentManager, discard SpawnManager)
    3.

    """

    @classmethod
    def default_config(cls):
        config = super(MARLSingleWaymoEnv, cls).default_config()
        config.update(MARL_SINGLE_WAYMO_ENV_CONFIG)
        config.register_type("randomized_dynamics", None, str)
        return config

    def setup_engine(self):
        self.in_stop = False

        # Call the setup_engine of BaseEnv
        self.engine.accept("r", self.reset)
        self.engine.accept("p", self.capture)
        # self.engine.register_manager("record_manager", RecordManager())
        # self.engine.register_manager("replay_manager", ReplayManager())

        self.engine.register_manager("data_manager", WaymoDataManager())
        self.engine.register_manager("map_manager", MAWaymoMapManager())

        # PZH: Stop using traffic manager!
        # if not self.config["no_traffic"]:
        #     if not self.config['replay']:
        #         self.engine.register_manager("traffic_manager", WaymoIDMTrafficManager())
        #     else:
        #         self.engine.register_manager("traffic_manager", WaymoTrafficManager())

        # PZH: Need to wait for the map to determine the spawn locations.
        # Therefore we need to put the init of agent manager later.

        # PZH: Overwrite the Agent Manager
        self.agent_manager = MAWaymoAgentManager(
            init_observations=self._get_observations(),
            init_action_space=self._get_action_space(),
            store_map=self.config["store_map"],
            store_map_buffer_size=self.config["store_map_buffer_size"]
        )
        self.engine.register_manager("agent_manager", self.agent_manager)

        # self.engine.register_manager("spawn_manager", MAWaymoSpawnManager())

        self.engine.accept("p", self.stop)
        self.engine.accept("q", self.switch_to_third_person_view)
        self.engine.accept("b", self.switch_to_top_down_view)

    def __init__(self, config):
        data_path = config.get("dataset_path", self.default_config()["dataset_path"])
        assert osp.exists(data_path), \
            "Can not find dataset: {}, Please download it from: " \
            "https://github.com/metadriverse/metadrive-scenario/releases.".format(data_path)

        config = copy.deepcopy(config)

        if config.get("discrete_action"):
            self._waymo_discrete_action = True
            config["discrete_action"] = False
        else:
            self._waymo_discrete_action = False

        config["waymo_data_directory"] = data_path
        super(MARLSingleWaymoEnv, self).__init__(config)
        self.agent_steps = 0
        self._sequential_seed = None

        self.dynamics_parameters_mean = None
        self.dynamics_parameters_std = None
        self.dynamics_function = None

    def set_dynamics_parameters_distribution(self, dynamics_parameters_mean=None, dynamics_parameters_std=None, dynamics_function=None):
        if dynamics_parameters_mean is not None:
            assert isinstance(dynamics_parameters_mean, np.ndarray)
        self.dynamics_parameters_mean = dynamics_parameters_mean
        self.dynamics_parameters_std = dynamics_parameters_std
        self.dynamics_function = dynamics_function

    def reset(self, *args, **kwargs):

        force_seed = None

        if self.config["randomized_dynamics"] == "naive":
            def _d(environment_seed=None, agent_name=None, latent_dict=None):
                s = np.random.randint(3)
                if s == 0:
                    return np.random.normal(-0.5, 0.2, size=5), {"mode": s}
                elif s == 1:
                    return np.random.normal(0, 0.2, size=5), {"mode": s}
                elif s == 2:
                    return np.random.normal(0.5, 0.2, size=5), {"mode": s}
                else:
                    raise ValueError()

            self.dynamics_function = _d

        finish = False

        while not finish:

            if self.config["sequential_seed"] and "force_seed" in kwargs:
                self._sequential_seed = kwargs["force_seed"]

            if not self.config["sequential_seed"] and "force_seed" in kwargs:
                force_seed = kwargs["force_seed"]

            if self.config["sequential_seed"]:
                if self._sequential_seed is None:
                    self._sequential_seed = self.config["start_case_index"]
                force_seed = self._sequential_seed
                self._sequential_seed += 1
                if self._sequential_seed >= self.config["start_case_index"] + self.config["case_num"]:
                    self._sequential_seed = self.config["start_case_index"]

            if self.config["randomized_dynamics"]:
                if not isinstance(self.agent_manager, MAWaymoAgentManager):
                    ret = super(MARLSingleWaymoEnv, self).reset(*args, **kwargs)

                assert isinstance(self.agent_manager, MAWaymoAgentManager)

                # For some reasons, env.reset could be called when the agent_manager is not set to
                # WaymoAgentManager yet.
                self.agent_manager.set_dynamics_parameters_distribution(
                    dynamics_parameters_mean=self.dynamics_parameters_mean,
                    dynamics_parameters_std=self.dynamics_parameters_std,
                    dynamics_function=self.dynamics_function,
                    latent_dict=self.latent_dict if hasattr(self, "latent_dict") else None
                )

            if "force_seed" in kwargs:
                kwargs.pop("force_seed")

            ret = super(MARLSingleWaymoEnv, self).reset(*args, force_seed=force_seed, **kwargs)

            # Since we are using Waymo real data, it is possible that the vehicle crashes with solid line already.
            # Remove those vehicles since it will terminate very soon during RL interaction.
            for agent_name, vehicle in self.agent_manager.active_agents.items():
                done = self._is_out_of_road(vehicle)
                if done:
                    self.agent_manager.put_to_static_list(vehicle)
                    if agent_name in ret:
                        ret.pop(agent_name)

            finish = len(self.vehicles) > 0

        self.agent_steps = 0
        self.dynamics_parameters_recorder = dict()
        return ret

    def set_dynamics_parameters(self, mean, std):
        assert self.config["lcf_dist"] == "normal"
        self.current_lcf_mean = mean
        self.current_lcf_std = std
        assert std > 0.0
        assert -1.0 <= self.current_lcf_mean <= 1.0

    def step(self, actions):

        # Let WaymoEnv to process action
        # This is equivalent to call MetaDriveEnv to process actions (it can deal with MA input!)
        obses, rewards, dones, infos = BaseEnv.step(self, actions)

        # Process agents according to whether they are done

        # def _after_vehicle_done(self, obs=None, reward=None, dones: dict = None, info=None):
        for v_id, v_info in infos.items():
            infos[v_id][TerminationState.MAX_STEP] = False
            if v_info.get("episode_length", 0) >= self.config["horizon"]:
                if dones[v_id] is not None:
                    infos[v_id][TerminationState.MAX_STEP] = True
                    dones[v_id] = True
                    self.dones[v_id] = True

            # Process a special case where RC > 1.
            # Even though we don't know what causes it, we should at least workaround it.
            if "route_completion" in infos[v_id]:
                rc = infos[v_id]["route_completion"]
                if (rc > 1.0 or rc < -0.1) and dones[v_id] is not None:
                    if rc > 1.0:
                        infos[v_id][TerminationState.SUCCESS] = True
                    dones[v_id] = True
                    self.dones[v_id] = True

        for dead_vehicle_id, done in dones.items():
            if done:
                self.agent_manager.finish(
                    dead_vehicle_id, ignore_delay_done=infos[dead_vehicle_id].get(TerminationState.SUCCESS, False)
                )
                self._update_camera_after_finish()
            # return obs, reward, dones, info

        # PZH: Do not respawn new vehicle!
        # Update respawn manager
        # if self.episode_step >= self.config["horizon"]:
        #     self.agent_manager.set_allow_respawn(False)
        # new_obs_dict, new_info_dict = self._respawn_vehicles(randomize_position=self.config["random_traffic"])
        # if new_obs_dict:
        #     for new_id, new_obs in new_obs_dict.items():
        #         o[new_id] = new_obs
        #         r[new_id] = 0.0
        #         i[new_id] = new_info_dict[new_id]
        #         d[new_id] = False

        self.agent_steps += len(self.vehicles)

        # Update __all__
        d_all = False
        if self.config["horizon"] is not None:  # No agent alive or a too long episode happens
            if self.episode_step >= self.config["horizon"]:
                d_all = True
        if len(self.vehicles) == 0:  # No agent alive
            d_all = True
        dones["__all__"] = d_all
        if dones["__all__"]:
            for k in dones.keys():
                dones[k] = True

        for k in infos.keys():
            infos[k]["agent_steps"] = self.agent_steps
            infos[k]["environment_seed"] = self.engine.global_seed
            infos[k]["vehicle_id"] = k

            # if dones[k]:
            #     infos[k]["raw_state"] = None
            #     infos[k]["dynamics"] = self.dynamics_parameters_recorder[k]
            # else:
            try:
                v = self.agent_manager.get_agent(k)
                # infos[k]["raw_state"] = v.get_raw_state()
                infos[k]["dynamics"] = v.get_dynamics_parameters()
                self.dynamics_parameters_recorder[k] = infos[k]["dynamics"]

            except (ValueError, KeyError):
                # infos[k]["raw_state"] = None
                if k in self.dynamics_parameters_recorder:
                    infos[k]["dynamics"] = self.dynamics_parameters_recorder[k]
                else:
                    infos[k]["dynamics"] = None

        return obses, rewards, dones, infos

    def _preprocess_actions(self, action):
        if self._waymo_discrete_action:
            new_action = {}
            discrete_action_dim = self.config["discrete_action_dim"]
            for k, v in action.items():
                assert 0 <= v < discrete_action_dim * discrete_action_dim
                a0 = v % discrete_action_dim
                a0 = _get_action(a0, discrete_action_dim)
                a1 = v // discrete_action_dim
                a1 = _get_action(a1, discrete_action_dim)
                new_action[k] = [a0, a1]
            action = new_action

        return super(MARLSingleWaymoEnv, self)._preprocess_actions(action)

    def _get_action_space(self):
        if self._waymo_discrete_action:
            from gym.spaces import Discrete
            discrete_action_dim = self.config["discrete_action_dim"]
            return {self.DEFAULT_AGENT: Discrete(discrete_action_dim * discrete_action_dim)}

        return {
            self.DEFAULT_AGENT: BaseVehicle.get_action_space_before_init(
                self.config["vehicle_config"]["extra_action_dim"], self.config["discrete_action"],
                self.config["discrete_steering_dim"], self.config["discrete_throttle_dim"]
            )
        }

    def _get_observations(self):
        return {self.DEFAULT_AGENT: self.get_single_observation(self.config["vehicle_config"])}

    def get_single_observation(self, vehicle_config):
        from newcopo.metadrive_scenario.marl_envs.observation import NewWaymoObservation
        return NewWaymoObservation(vehicle_config)

    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        raise ValueError("check marl_waymo_env.py")
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # update obs
        self.observations[vehicle_id].lateral_dist = \
            self.engine.map_manager.current_routes[vehicle_id].local_coordinates(vehicle.position)[-1]

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral_reward"]:
            lateral_factor = clip(1 - 2 * abs(lateral_now) / 6, 0.0, 1.0)
        else:
            lateral_factor = 1.0

        reward = 0
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor

        step_info["step_reward"] = reward

        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]

        step_info["track_length"] = vehicle.navigation.reference_trajectory.length
        step_info["current_distance"] = vehicle.navigation.reference_trajectory.local_coordinates(vehicle.position)[0]
        rc = step_info["current_distance"] / step_info["track_length"]
        step_info["route_completion"] = rc

        # print("Vehicle {} Track Length {:.3f} Current Dis {:.3f} Route Completion {:.4f}".format(
        #     vehicle, step_info["track_length"] , step_info["current_distance"] , step_info["route_completion"]
        # ))

        return reward, step_info

    # def _is_arrive_destination(self, vehicle):
    #     if np.linalg.norm(vehicle.position - self.engine.map_manager.dest_points[vehicle.name]) < 5:
    #         return True
    #     else:
    #         return False

    def _is_arrive_destination(self, vehicle):
        long, lat = vehicle.navigation.reference_trajectory.local_coordinates(vehicle.position)

        total_length = vehicle.navigation.reference_trajectory.length
        current_distance = long

        # agent_name = self.agent_manager.object_to_agent(vehicle.name)
        # threshold = 5

        # if np.linalg.norm(vehicle.position - self.engine.map_manager.dest_points[agent_name]) < threshold:
        #     return True
        # elif current_distance + threshold > total_length:  # Route Completion ~= 1.0
        #     return True
        # else:
        #     return False

        # Update 2022-02-05: Use RC as the only criterion to determine arrival.
        route_completion = current_distance / total_length
        if route_completion > 0.95:  # Route Completion ~= 1.0
            return True
        else:
            return False

    def _is_out_of_road(self, vehicle):
        # A specified function to determine whether this vehicle should be done.

        if self.config["relax_out_of_road_done"]:
            agent_name = self.agent_manager.object_to_agent(vehicle.name)
            lat = abs(self.observations[agent_name].lateral_dist)
            done = lat > 10
            done = done or vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line
            return done

        done = vehicle.crash_sidewalk or vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line
        if self.config["out_of_route_done"]:
            agent_name = self.agent_manager.object_to_agent(vehicle.name)
            done = done or abs(self.observations[agent_name].lateral_dist) > 10
        return done


class MARLWaymoEnv():
    """
    PZH Note:
    We should strictly spawn N agents at N locations, which are the initial positions of all
    "valid" vehicles in one real waymo scene.

    Problems:
    1. How to determine a vehicles in the scene is valid?
    2. How to spawn RL vehicles to each valid vehicle? (Modify TrafficManager, AgentManager, discard SpawnManager)
    3.

    """
    @classmethod
    def default_config(cls):
        config = super(MARLWaymoEnv, cls).default_config()
        config.update(MARL_WAYMO_ENV_CONFIG)
        return config

    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]

        lateral_now = 3  # Stupid fix
        long_now = long_last = 0
        if current_lane is not None:
            long_last, _ = current_lane.local_coordinates(vehicle.last_position)
            long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # update obs
        self.observations[vehicle_id].lateral_dist = \
            self.engine.map_manager.current_routes[vehicle_id].local_coordinates(vehicle.position)[-1]

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral_reward"]:
            lateral_factor = clip(1 - 2 * abs(lateral_now) / 6, 0.0, 1.0)
        else:
            lateral_factor = 1.0

        reward = 0
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor

        step_info["step_reward"] = reward

        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]

        step_info["track_length"] = vehicle.navigation.reference_trajectory.length
        step_info["current_distance"] = vehicle.navigation.reference_trajectory.local_coordinates(vehicle.position)[0]
        rc = step_info["current_distance"] / step_info["track_length"]
        step_info["route_completion"] = rc

        step_info["carsize"] = [vehicle.WIDTH / 10, vehicle.LENGTH / 10]

        # Compute state difference metrics
        data = self.engine.data_manager.get_case(self.engine.global_seed)
        agent_xy = vehicle.position
        if vehicle_id == "sdc":
            native_vid = data["sdc_index"]
        else:
            native_vid = vehicle_id

        if native_vid in data["tracks"] and len(data["tracks"][native_vid]["state"]) > 0:
            expert_state_list = data["tracks"][native_vid]["state"]
            mask = expert_state_list[:, -1]
            largest_valid_index = np.max(np.where(expert_state_list[:, -1] == 1)[0])

            if self.episode_step > largest_valid_index:
                current_step = largest_valid_index
            else:
                current_step = self.episode_step

            while mask[current_step] == 0.0:
                current_step -= 1
                if current_step == 0:
                    break

            expert_state = expert_state_list[current_step]
            expert_xy = waymo_2_metadrive_position(expert_state[:2])
            dist = np.linalg.norm(agent_xy - expert_xy)
            step_info["distance_error"] = dist

            last_state = expert_state_list[largest_valid_index]
            last_expert_xy = waymo_2_metadrive_position(last_state[:2])
            last_dist = np.linalg.norm(agent_xy - last_expert_xy)
            step_info["distance_error_final"] = last_dist

            reward = reward - self.config["distance_penalty"] * dist

        # print("Vehicle {} Track Length {:.3f} Current Dis {:.3f} Route Completion {:.4f}".format(
        #     vehicle, step_info["track_length"] , step_info["current_distance"] , step_info["route_completion"]
        # ))

        if hasattr(vehicle, "_dynamics_mode"):
            step_info["dynamics_mode"] = vehicle._dynamics_mode

        return reward, step_info


if __name__ == "__main__":
    # video_name = "MARL_WAYMO_ENV.mp4"
    # tmp_folder = video_name + ".TMP"
    # os.makedirs(tmp_folder, exist_ok=True)
    env = MARLWaymoEnv(dict(
        # randomized_dynamics="naive",

        # relax_out_of_road_done=True,
        # "save_memory": True,
        # "save_memory_max_len": 2,
        # "discrete_action": True,
        # "discrete_action_dim": 5
    ))

    frame_count = 0
    for ep in tqdm(range(100), desc="Episode"):
        # env.reset(force_seed=ep)
        env.reset()
        for t in tqdm(range(1000), desc="Step"):
            o, r, d, i = env.step({key: [1, 1] for key in env.vehicles.keys()})

            assert env.observation_space.contains(o)

            # o, r, d, i = env.step(env.action_space.sample())

            # print({k: ii["distance_error"] for k, ii in i.items() if "distance_error" in ii})
            # print({k for k, ii in i.items() if "distance_error" not in ii})

            if d["__all__"]:
                break
                # env.reset()
            # os.environ['SDL_VIDEODRIVER'] = 'dummy'
            # import pygame
            #
            # ret = env.render(
            #     mode="top_down", film_size=(3000, 3000), screen_size=(3000, 3000), track_target_vehicle=False
            # )
            # pygame.image.save(ret, "{}/{}.png".format(tmp_folder, frame_count))
            frame_count += 1
    # image_to_video(video_name, tmp_folder)
    # shutil.rmtree(tmp_folder)