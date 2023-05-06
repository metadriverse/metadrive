import logging
import os
import tempfile
from dataclasses import dataclass
from os.path import join

import numpy as np
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString

from metadrive.scenario import ScenarioDescription as SD
from metadrive.type import MetaDriveType
from metadrive.utils.coordinates_shift import nuplan_to_metadrive_vector
from metadrive.utils.math import compute_angular_velocity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from metadrive.utils import is_win

try:
    import geopandas as gpd
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
    from shapely.ops import unary_union
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
    import hydra
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
    from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
    from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
    from nuplan.planning.script.utils import set_up_common_builder
    import nuplan
except ImportError:
    logger.warning("Can not import nuplan-devkit")

logger.warning(
    "This converters will de deprecated. "
    "Use tools in ScenarioNet instead: https://github.com/metadriverse/ScenarioNet."
)

NUPLAN_PACKAGE_PATH = os.path.dirname(nuplan.__file__)
EGO = "ego"


def get_nuplan_scenarios(dataset_parameters, nuplan_package_path=NUPLAN_PACKAGE_PATH):
    """
    Return a list of nuplan scenarios according to dataset_parameters
    """
    base_config_path = os.path.join(nuplan_package_path, "planning", "script")
    simulation_hydra_paths = construct_simulation_hydra_paths(base_config_path)

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize_config_dir(config_dir=simulation_hydra_paths.config_path)

    save_dir = tempfile.mkdtemp()
    ego_controller = 'perfect_tracking_controller'  # [log_play_back_controller, perfect_tracking_controller]
    observation = 'box_observation'  # [box_observation, idm_agents_observation, lidar_pc_observation]

    # Compose the configuration
    overrides = [
        f'group={save_dir}',
        'worker=sequential',
        f'ego_controller={ego_controller}',
        f'observation={observation}',
        f'hydra.searchpath=[{simulation_hydra_paths.common_dir}, {simulation_hydra_paths.experiment_dir}]',
        'output_dir=${group}/${experiment}',
        'metric_dir=${group}/${experiment}',
        *dataset_parameters,
    ]
    overrides.extend(
        [
            f'job_name=planner_tutorial', 'experiment=${experiment_name}/${job_name}',
            f'experiment_name=planner_tutorial'
        ]
    )

    # get config
    cfg = hydra.compose(config_name=simulation_hydra_paths.config_name, overrides=overrides)

    profiler_name = 'building_simulation'
    common_builder = set_up_common_builder(cfg=cfg, profiler_name=profiler_name)

    # Build scenario builder
    scenario_builder = build_scenario_builder(cfg=cfg)
    scenario_filter = build_scenario_filter(cfg.scenario_filter)

    # get scenarios from database
    return scenario_builder.get_scenarios(scenario_filter, common_builder.worker)


def construct_simulation_hydra_paths(base_config_path: str):
    """
    Specifies relative paths to simulation configs to pass to hydra to declutter tutorial.
    :param base_config_path: Base config path.
    :return: Hydra config path.
    """
    common_dir = "file://" + join(base_config_path, 'config', 'common')
    config_name = 'default_simulation'
    config_path = join(base_config_path, 'config', 'simulation')
    experiment_dir = "file://" + join(base_config_path, 'experiments')
    return HydraConfigPaths(common_dir, config_name, config_path, experiment_dir)


@dataclass
class HydraConfigPaths:
    """
    Stores relative hydra paths to declutter tutorial.
    """

    common_dir: str
    config_name: str
    config_path: str
    experiment_dir: str


def extract_centerline(map_obj, nuplan_center):
    path = map_obj.baseline_path.discrete_path
    points = np.array([nuplan_to_metadrive_vector([pose.x, pose.y], nuplan_center) for pose in path])
    return points


def get_points_from_boundary(boundary, center):
    path = boundary.discrete_path
    points = [(pose.x, pose.y) for pose in path]
    points = nuplan_to_metadrive_vector(points, center)
    return points


def get_line_type(nuplan_type):
    return MetaDriveType.LINE_BROKEN_SINGLE_WHITE
    # Always return broken line type
    if nuplan_type == 2:
        return MetaDriveType.LINE_SOLID_SINGLE_WHITE
    elif nuplan_type == 0:
        return MetaDriveType.LINE_BROKEN_SINGLE_WHITE
    elif nuplan_type == 3:
        return MetaDriveType.LINE_UNKNOWN
    else:
        raise ValueError("Unknown line tyep: {}".format(nuplan_type))


def extract_map_features(map_api, center, radius=250):
    ret = {}
    np.seterr(all='ignore')
    # Center is Important !
    layer_names = [
        SemanticMapLayer.LANE_CONNECTOR,
        SemanticMapLayer.LANE,
        SemanticMapLayer.CROSSWALK,
        SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.STOP_LINE,
        SemanticMapLayer.WALKWAYS,
        SemanticMapLayer.CARPARK_AREA,
        SemanticMapLayer.ROADBLOCK,
        SemanticMapLayer.ROADBLOCK_CONNECTOR,

        # unsupported yet
        # SemanticMapLayer.STOP_SIGN,
        # SemanticMapLayer.DRIVABLE_AREA,
    ]
    center_for_query = Point2D(*center)
    nearest_vector_map = map_api.get_proximal_map_objects(center_for_query, radius, layer_names)
    boundaries = map_api._get_vector_map_layer(SemanticMapLayer.BOUNDARIES)
    # Filter out stop polygons in turn stop
    if SemanticMapLayer.STOP_LINE in nearest_vector_map:
        stop_polygons = nearest_vector_map[SemanticMapLayer.STOP_LINE]
        nearest_vector_map[SemanticMapLayer.STOP_LINE] = [
            stop_polygon for stop_polygon in stop_polygons if stop_polygon.stop_line_type != StopLineType.TURN_STOP
        ]
    block_polygons = []
    for layer in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
        for block in nearest_vector_map[layer]:
            for lane_meta_data in block.interior_edges:
                if not hasattr(lane_meta_data, "baseline_path"):
                    continue
                if isinstance(lane_meta_data.polygon.boundary, MultiLineString):
                    # logger.warning("Stop using boundaries! Use exterior instead!")
                    boundary = gpd.GeoSeries(lane_meta_data.polygon.boundary).explode(index_parts=True)
                    sizes = []
                    for idx, polygon in enumerate(boundary[0]):
                        sizes.append(len(polygon.xy[1]))
                    points = boundary[0][np.argmax(sizes)].xy
                elif isinstance(lane_meta_data.polygon.boundary, LineString):
                    points = lane_meta_data.polygon.boundary.xy
                polygon = [[points[0][i], points[1][i]] for i in range(len(points[0]))]
                polygon = nuplan_to_metadrive_vector(polygon, nuplan_center=[center[0], center[1]])
                ret[lane_meta_data.id] = {
                    SD.TYPE: MetaDriveType.LANE_SURFACE_STREET,
                    SD.POLYLINE: extract_centerline(lane_meta_data, center),
                    SD.POLYGON: polygon
                }
                if layer == SemanticMapLayer.ROADBLOCK_CONNECTOR:
                    continue
                left = lane_meta_data.left_boundary
                if left.id not in ret:
                    line_type = get_line_type(int(boundaries.loc[[str(left.id)]]["boundary_type_fid"]))
                    if line_type != MetaDriveType.LINE_UNKNOWN:
                        ret[left.id] = {SD.TYPE: line_type, SD.POLYLINE: get_points_from_boundary(left, center)}

            if layer == SemanticMapLayer.ROADBLOCK:
                block_polygons.append(block.polygon)

    interpolygons = [block.polygon for block in nearest_vector_map[SemanticMapLayer.INTERSECTION]]
    # logger.warning("Stop using boundaries! Use exterior instead!")
    boundaries = gpd.GeoSeries(unary_union(interpolygons + block_polygons)).boundary.explode(index_parts=True)
    # boundaries.plot()
    # plt.show()
    for idx, boundary in enumerate(boundaries[0]):
        block_points = np.array(list(i for i in zip(boundary.coords.xy[0], boundary.coords.xy[1])))
        block_points = nuplan_to_metadrive_vector(block_points, center)
        id = "boundary_{}".format(idx)
        ret[id] = {SD.TYPE: MetaDriveType.LINE_SOLID_SINGLE_WHITE, SD.POLYLINE: block_points}
    np.seterr(all='warn')
    return ret


def set_light_status(status):
    if status == TrafficLightStatusType.GREEN:
        return MetaDriveType.LIGHT_GREEN
    elif status == TrafficLightStatusType.RED:
        return MetaDriveType.LIGHT_RED
    elif status == TrafficLightStatusType.YELLOW:
        return MetaDriveType.LIGHT_YELLOW
    elif status == TrafficLightStatusType.UNKNOWN:
        return MetaDriveType.LIGHT_UNKNOWN


def set_light_position(scenario, lane_id, center, target_position=8):
    lane = scenario.map_api.get_map_object(str(lane_id), SemanticMapLayer.LANE_CONNECTOR)
    assert lane is not None, "Can not find lane: {}".format(lane_id)
    path = lane.baseline_path.discrete_path
    acc_length = 0
    point = [path[0].x, path[0].y]
    for k, point in enumerate(path[1:], start=1):
        previous_p = path[k - 1]
        acc_length += np.linalg.norm([point.x - previous_p.x, point.y - previous_p.y])
        if acc_length > target_position:
            break
    return [point.x - center[0], point.y - center[1]]


def extract_traffic_light(scenario, center):
    length = scenario.get_number_of_iterations()

    frames = [
        {str(t.lane_connector_id): t.status
         for t in scenario.get_traffic_light_status_at_iteration(i)} for i in range(length)
    ]
    all_lights = set()
    for frame in frames:
        all_lights.update(frame.keys())

    lights = {
        k: {
            "type": MetaDriveType.TRAFFIC_LIGHT,
            "state": {
                SD.TRAFFIC_LIGHT_STATUS: [MetaDriveType.LIGHT_UNKNOWN] * length
            },
            SD.TRAFFIC_LIGHT_POSITION: None,
            SD.TRAFFIC_LIGHT_LANE: str(k),
            "metadata": dict(track_length=length, type=None, object_id=str(k), lane_id=str(k), dataset="nuplan")
        }
        for k in list(all_lights)
    }

    for k, frame in enumerate(frames):
        for lane_id, status in frame.items():
            lane_id = str(lane_id)
            lights[lane_id]["state"][SD.TRAFFIC_LIGHT_STATUS][k] = set_light_status(status)
            if lights[lane_id][SD.TRAFFIC_LIGHT_POSITION] is None:
                assert isinstance(lane_id, str), "Lane ID should be str"
                lights[lane_id][SD.TRAFFIC_LIGHT_POSITION] = set_light_position(scenario, lane_id, center)
                lights[lane_id][SD.METADATA][SD.TYPE] = MetaDriveType.TRAFFIC_LIGHT

    return lights


def get_traffic_obj_type(nuplan_type):
    if nuplan_type == TrackedObjectType.VEHICLE:
        return MetaDriveType.VEHICLE
    elif nuplan_type == TrackedObjectType.TRAFFIC_CONE:
        return MetaDriveType.TRAFFIC_CONE
    elif nuplan_type == TrackedObjectType.PEDESTRIAN:
        return MetaDriveType.PEDESTRIAN
    elif nuplan_type == TrackedObjectType.BICYCLE:
        return MetaDriveType.CYCLIST
    elif nuplan_type == TrackedObjectType.BARRIER:
        return MetaDriveType.TRAFFIC_BARRIER
    elif nuplan_type == TrackedObjectType.EGO:
        raise ValueError("Ego should not be in detected resukts")
    else:
        return None


def parse_object_state(obj_state, nuplan_center):
    ret = {}
    ret["position"] = nuplan_to_metadrive_vector([obj_state.center.x, obj_state.center.y], nuplan_center)
    ret["heading"] = obj_state.center.heading
    ret["velocity"] = nuplan_to_metadrive_vector([obj_state.velocity.x, obj_state.velocity.y])
    ret["valid"] = 1
    ret["length"] = obj_state.box.length
    ret["width"] = obj_state.box.width
    ret["height"] = obj_state.box.height
    return ret


def parse_ego_vehicle_state(state, nuplan_center):
    center = nuplan_center
    ret = {}
    ret["position"] = nuplan_to_metadrive_vector([state.waypoint.x, state.waypoint.y], center)
    ret["heading"] = state.waypoint.heading
    ret["velocity"] = nuplan_to_metadrive_vector([state.agent.velocity.x, state.agent.velocity.y])
    ret["angular_velocity"] = state.dynamic_car_state.angular_velocity
    ret["valid"] = 1
    ret["length"] = state.agent.box.length
    ret["width"] = state.agent.box.width
    ret["height"] = state.agent.box.height
    return ret


def parse_ego_vehicle_state_trajectory(scenario, nuplan_center):
    data = [
        parse_ego_vehicle_state(scenario.get_ego_state_at_iteration(i), nuplan_center)
        for i in range(scenario.get_number_of_iterations())
    ]
    for i in range(len(data) - 1):
        data[i]["angular_velocity"] = compute_angular_velocity(
            initial_heading=data[i]["heading"], final_heading=data[i + 1]["heading"], dt=scenario.database_interval
        )
    return data


def extract_traffic(scenario: NuPlanScenario, center):
    episode_len = scenario.get_number_of_iterations()
    detection_ret = []
    all_objs = set()
    all_objs.add(EGO)
    for frame_data in [scenario.get_tracked_objects_at_iteration(i).tracked_objects for i in range(episode_len)]:
        new_frame_data = {}
        for obj in frame_data:
            new_frame_data[obj.track_token] = obj
            all_objs.add(obj.track_token)
        detection_ret.append(new_frame_data)

    tracks = {
        k: dict(
            type=MetaDriveType.UNSET,
            state=dict(
                position=np.zeros(shape=(episode_len, 3)),
                heading=np.zeros(shape=(episode_len, )),
                velocity=np.zeros(shape=(episode_len, 2)),
                valid=np.zeros(shape=(episode_len, )),
                length=np.zeros(shape=(episode_len, 1)),
                width=np.zeros(shape=(episode_len, 1)),
                height=np.zeros(shape=(episode_len, 1))
            ),
            metadata=dict(track_length=episode_len, nuplan_type=None, type=None, object_id=k, nuplan_id=k)
        )
        for k in list(all_objs)
    }

    tracks_to_remove = set()

    for frame_idx, frame in enumerate(detection_ret):
        for nuplan_id, obj_state, in frame.items():
            assert isinstance(obj_state, Agent) or isinstance(obj_state, StaticObject)
            obj_type = get_traffic_obj_type(obj_state.tracked_object_type)
            if obj_type is None:
                tracks_to_remove.add(nuplan_id)
                continue
            tracks[nuplan_id][SD.TYPE] = obj_type
            if tracks[nuplan_id][SD.METADATA]["nuplan_type"] is None:
                tracks[nuplan_id][SD.METADATA]["nuplan_type"] = int(obj_state.tracked_object_type)
                tracks[nuplan_id][SD.METADATA]["type"] = obj_type

            state = parse_object_state(obj_state, center)
            tracks[nuplan_id]["state"]["position"][frame_idx] = [state["position"][0], state["position"][1], 0.0]
            tracks[nuplan_id]["state"]["heading"][frame_idx] = state["heading"]
            tracks[nuplan_id]["state"]["velocity"][frame_idx] = state["velocity"]
            tracks[nuplan_id]["state"]["valid"][frame_idx] = 1
            tracks[nuplan_id]["state"]["length"][frame_idx] = state["length"]
            tracks[nuplan_id]["state"]["width"][frame_idx] = state["width"]
            tracks[nuplan_id]["state"]["height"][frame_idx] = state["height"]

    for track in list(tracks_to_remove):
        tracks.pop(track)

    # ego
    sdc_traj = parse_ego_vehicle_state_trajectory(scenario, center)
    ego_track = tracks[EGO]

    for frame_idx, obj_state in enumerate(sdc_traj):
        obj_type = MetaDriveType.VEHICLE
        ego_track[SD.TYPE] = obj_type
        if ego_track[SD.METADATA]["nuplan_type"] is None:
            ego_track[SD.METADATA]["nuplan_type"] = int(TrackedObjectType.EGO)
            ego_track[SD.METADATA]["type"] = obj_type
        state = obj_state
        ego_track["state"]["position"][frame_idx] = [state["position"][0], state["position"][1], 0.0]
        ego_track["state"]["valid"][frame_idx] = 1
        ego_track["state"]["heading"][frame_idx] = state["heading"]
        # this velocity is in ego car frame, abort
        # ego_track["state"]["velocity"][frame_idx] = state["velocity"]

        ego_track["state"]["length"][frame_idx] = state["length"]
        ego_track["state"]["width"][frame_idx] = state["width"]
        ego_track["state"]["height"][frame_idx] = state["height"]

    # get velocity here
    vel = ego_track["state"]["position"][1:] - ego_track["state"]["position"][:-1]
    ego_track["state"]["velocity"][:-1] = vel[..., :2] / 0.1
    ego_track["state"]["velocity"][-1] = ego_track["state"]["velocity"][-2]

    # check
    assert EGO in tracks
    for track_id in tracks:
        assert tracks[track_id][SD.TYPE] != MetaDriveType.UNSET

    return tracks


def convert_one_scenario(scenario: NuPlanScenario):
    """
    Data will be interpolated to 0.1s time interval, while the time interval of original key frames are 0.5s.
    """
    scenario_log_interval = scenario.database_interval
    assert abs(scenario_log_interval - 0.1) < 1e-3, "Log interval should be 0.1 or Interpolating is required! " \
                                                    "By setting NuPlan subsample ratio can address this"

    result = SD()
    result[SD.ID] = scenario.scenario_name
    result[SD.VERSION] = "nuplan" + scenario.map_version
    result[SD.LENGTH] = scenario.get_number_of_iterations()
    # metadata
    result[SD.METADATA] = {}
    result[SD.METADATA]["dataset"] = "nuplan"
    result[SD.METADATA]["map"] = scenario.map_api.map_name
    result[SD.METADATA][SD.METADRIVE_PROCESSED] = False
    result[SD.METADATA]["map_version"] = scenario.map_version
    result[SD.METADATA]["log_name"] = scenario.log_name
    result[SD.METADATA]["scenario_extraction_info"] = scenario._scenario_extraction_info.__dict__
    result[SD.METADATA]["ego_vehicle_parameters"] = scenario.ego_vehicle_parameters.__dict__
    result[SD.METADATA]["coordinate"] = "right-handed"
    result[SD.METADATA]["scenario_token"] = scenario.scenario_name
    result[SD.METADATA]["scenario_id"] = scenario.scenario_name
    result[SD.METADATA]["scenario_type"] = scenario.scenario_type
    result[SD.METADATA]["sample_rate"] = scenario_log_interval
    result[SD.METADATA][SD.TIMESTEP] = np.asarray([i * scenario_log_interval for i in range(result[SD.LENGTH])])

    # centered all positions to ego car
    state = scenario.get_ego_state_at_iteration(0)
    scenario_center = [state.waypoint.x, state.waypoint.y]

    result[SD.TRACKS] = extract_traffic(scenario, scenario_center)
    result[SD.METADATA][SD.SDC_ID] = EGO

    # traffic light
    result[SD.DYNAMIC_MAP_STATES] = extract_traffic_light(scenario, scenario_center)

    # map
    result[SD.MAP_FEATURES] = extract_map_features(scenario.map_api, scenario_center)

    return result
