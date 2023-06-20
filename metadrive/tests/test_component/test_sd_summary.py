import inspect
import os
from typing import Callable
from metadrive.type import MetaDriveType
from metadrive.engine.asset_loader import AssetLoader
from metadrive.scenario.scenario_description import ScenarioDescription as SD
from metadrive.scenario.utils import read_scenario_data, read_dataset_summary


def test_read_data_and_create_summary():
    # waymo = AssetLoader.file_path("waymo", return_raw_style=False)
    nuscenes = AssetLoader.file_path("nuscenes", return_raw_style=False)
    summary, scenarios, mapping = read_dataset_summary(nuscenes)
    sd_scenario = read_scenario_data(os.path.join(nuscenes, mapping[scenarios[2]], scenarios[2]))

    summary_dict = {}
    for track_id, track in sd_scenario[SD.TRACKS].items():
        summary_dict[track_id] = SD.get_object_summary(state_dict=track, id=track_id)
    sd_scenario[SD.METADATA][SD.SUMMARY.OBJECT_SUMMARY] = summary_dict

    # test
    assert sd_scenario[SD.METADATA][SD.SDC_ID] in summary_dict

    # count some objects occurrence
    sd_scenario[SD.METADATA][SD.SUMMARY.NUMBER_SUMMARY] = SD.get_number_summary(sd_scenario)

    all_v = sd_scenario[SD.METADATA][SD.SUMMARY.NUMBER_SUMMARY][SD.SUMMARY.NUM_OBJECTS_EACH_TYPE][MetaDriveType.VEHICLE]
    moving_v = sd_scenario[SD.METADATA][SD.SUMMARY.NUMBER_SUMMARY][SD.SUMMARY.NUM_MOVING_OBJECTS_EACH_TYPE][
        MetaDriveType.VEHICLE]
    static_v = all_v - moving_v
    assert static_v == 14, "Wrong calculation"


def test_repeated_key_check():
    def _check_dict_variable(d):
        attributes = {key: value for key, value in d.__dict__.items() if not key.startswith('__') and not callable(key)}
        keys = set()
        values = set()
        for k, v in attributes.items():
            if isinstance(v, str):
                keys.add(k)
                values.add(v)
        assert len(keys) == len(values)

    _check_dict_variable(SD)
    _check_dict_variable(SD.SUMMARY)


if __name__ == '__main__':
    # test_repeated_key_check()
    test_read_data_and_create_summary()
