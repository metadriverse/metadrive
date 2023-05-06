"""
This script aims to convert nuplan scenarios to ScenarioDescription, so that we can load any nuplan scenarios into
MetaDrive.
"""
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import CameraChannel, LidarChannel

from metadrive.utils.nuplan.utils import get_nuplan_scenarios

# import os
#
# os.environ["NUPLAN_DATA_STORE"] = "s3"

import logging

logger = logging.getLogger(__name__)

logger.warning(
    "This converters will de deprecated. "
    "Use tools in ScenarioNet instead: https://github.com/metadriverse/ScenarioNet."
)

if __name__ == '__main__':
    raise ValueError("It seems the sensor data is not ready yet")
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario, CameraChannel, LidarChannel
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo

    NUPLAN_DATA_ROOT = "/mnt/data/nuplan/dataset/"
    NUPLAN_MAP_VERSION = "nuplan-maps-v1.0"
    NUPLAN_MAPS_ROOT = "/mnt/data/nuplan/dataset/maps"
    NUPLAN_SENSOR_ROOT = f"/mnt/data/nuplan/dataset/nuplan-v1.1/sensor_blobs"
    TEST_DB_FILE = f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini/2021.05.12.22.00.38_veh-35_01008_01518.db"
    MAP_NAME = "us-nv-las-vegas"
    TEST_INITIAL_LIDAR_PC = "58ccd3df9eab54a3"
    TEST_INITIAL_TIMESTAMP = 1620858198150622

    scenario = NuPlanScenario(
        data_root=f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini",
        log_file_load_path=TEST_DB_FILE,
        initial_lidar_token=TEST_INITIAL_LIDAR_PC,
        initial_lidar_timestamp=TEST_INITIAL_TIMESTAMP,
        scenario_type="scenario_type",
        map_root=NUPLAN_MAPS_ROOT,
        map_version=NUPLAN_MAP_VERSION,
        map_name=MAP_NAME,
        scenario_extraction_info=ScenarioExtractionInfo(
            scenario_name="scenario_name", scenario_duration=20, extraction_offset=1, subsample_ratio=0.5
        ),
        ego_vehicle_parameters=get_pacifica_parameters(),
        sensor_root=NUPLAN_DATA_ROOT + "/nuplan-v1.1/sensor_blobs",
    )
