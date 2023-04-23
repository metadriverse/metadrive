import logging
import os
import tempfile
from dataclasses import dataclass
from os.path import join
from metadrive.type import MetaDriveType
import numpy as np
import tqdm
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString

from metadrive.scenario import ScenarioDescription as SD
from metadrive.utils.coordinates_shift import nuplan_to_metadrive_vector

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

NUPLAN_PACKAGE_PATH = os.path.dirname(nuplan.__file__)


def extract_centerline(map_obj, nuplan_center):
    path = map_obj.baseline_path.discrete_path
    points = np.array([nuplan_to_metadrive_vector([pose.x, pose.y], nuplan_center) for pose in path])
    return points


def get_map_features(map_api, center, radius=250):
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
    # Filter out stop polygons in turn stop
    if SemanticMapLayer.STOP_LINE in nearest_vector_map:
        stop_polygons = nearest_vector_map[SemanticMapLayer.STOP_LINE]
        nearest_vector_map[SemanticMapLayer.STOP_LINE] = [
            stop_polygon for stop_polygon in stop_polygons if stop_polygon.stop_line_type != StopLineType.TURN_STOP
        ]

    block_polygons = []
    name = {SemanticMapLayer.ROADBLOCK: "Road Block", SemanticMapLayer.ROADBLOCK_CONNECTOR: "Road Connector"}

    for layer in tqdm.tqdm([SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]):
        for block in tqdm.tqdm(nearest_vector_map[layer], leave=False, desc="Building {}".format(name[layer])):
            for lane_meta_data in block.interior_edges:
                if hasattr(lane_meta_data, "baseline_path"):
                    if isinstance(lane_meta_data.polygon.boundary, MultiLineString):
                        boundary = gpd.GeoSeries(lane_meta_data.polygon.boundary).explode(index_parts=True)
                        sizes = []
                        for idx, polygon in enumerate(boundary[0]):
                            sizes.append(len(polygon.xy[1]))
                        points = boundary[0][np.argmax(sizes)].xy
                    elif isinstance(lane_meta_data.polygon.boundary, LineString):
                        points = lane_meta_data.polygon.boundary.xy
                    polygon = [[points[0][i], points[1][i]] for i in range(len(points[0]))]
                    polygon = nuplan_to_metadrive_vector(polygon, nuplan_center=[center[0], center[1]])
                    ret[lane_meta_data.id] = {SD.TYPE: MetaDriveType.LANE_SURFACE_STREET,
                                              SD.POLYLINE: extract_centerline(lane_meta_data, center),
                                              SD.POLYGON: polygon}

            if layer == SemanticMapLayer.ROADBLOCK:
                block_polygons.append(block.polygon)

    interpolygons = [block.polygon for block in nearest_vector_map[SemanticMapLayer.INTERSECTION]]
    boundaries = gpd.GeoSeries(unary_union(interpolygons + block_polygons)).boundary.explode(index_parts=True)
    # boundaries.plot()
    # plt.show()
    for idx, boundary in enumerate(boundaries[0]):
        block_points = np.array(list(i for i in zip(boundary.coords.xy[0], boundary.coords.xy[1])))
        block_points = nuplan_to_metadrive_vector(block_points, center)
        id = "boundary_{}".format(idx)
        ret[id] = {SD.TYPE: MetaDriveType.LINE_SOLID_SINGLE_WHITE,
                   SD.POLYLINE: block_points}
    np.seterr(all='warn')
    return ret


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
    result[SD.METADATA]["map_version"] = scenario.map_version
    result[SD.METADATA]["log_name"] = scenario.log_name
    result[SD.METADATA]["scenario_extraction_info"] = scenario._scenario_extraction_info
    result[SD.METADATA]["ego_vehicle_parameters"] = scenario.ego_vehicle_parameters
    result[SD.METADATA]["coordinate"] = "right-handed"
    result[SD.METADATA]["scenario_token"] = scenario.scenario_name
    result[SD.METADATA]["scenario_id"] = scenario.scenario_name
    result[SD.METADATA]["scenario_type"] = scenario.scenario_type
    result[SD.METADATA]["sample_rate"] = scenario_log_interval
    result[SD.METADATA][SD.TIMESTEP] = [i * scenario_log_interval for i in range(result[SD.LENGTH])]

    # centered all positions to ego car
    state = scenario.get_ego_state_at_iteration(0)
    scenario_center = [state.waypoint.x, state.waypoint.y]

    # map
    result[SD.MAP_FEATURES] = get_map_features(scenario.map_api, scenario_center)

    result[SD.TRACKS] = None
    result[SD.METADATA][SD.SDC_ID] = "ego"

    result[SD.DYNAMIC_MAP_STATES] = {}

    return result


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
        *dataset_parameters,
    ]
    if is_win():
        overrides.extend(
            [
                f'job_name=planner_tutorial',
                'experiment=${experiment_name}/${job_name}/${experiment_time}',
            ]
        )
    else:
        overrides.append(f'experiment_name=planner_tutorial')

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
