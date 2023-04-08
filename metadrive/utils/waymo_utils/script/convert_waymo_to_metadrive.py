"""
This script takes --folder as input. It is the folder storing a batch of tfrecord file.
This script will create the output folder "processed_data" sharing the same level as `--folder`.

-- folder
-- processed_data

"""
import argparse
import copy
import os
import pickle

import numpy as np
from tqdm import tqdm

from metadrive.constants import DATA_VERSION

try:
    import tensorflow as tf
except ImportError:
    pass
from metadrive.utils.waymo_utils.protos import scenario_pb2
from metadrive.scenario import ScenarioDescription as SD
from metadrive.type import MetaDriveType
from metadrive.utils.waymo_utils.utils import extract_tracks, extract_dynamic_map_states, extract_map_features, compute_width
import sys


def validate_sdc_track(sdc_state):
    """
    This function filters the scenario based on SDC information.

    Rule 1: Filter out if the trajectory length < 10

    Rule 2: Filter out if the whole trajectory last < 5s, assuming sampling frequency = 10Hz.
    """
    valid_array = sdc_state["valid"]
    sdc_trajectory = sdc_state["position"][valid_array, :2]
    sdc_track_length = [
        np.linalg.norm(sdc_trajectory[i] - sdc_trajectory[i + 1]) for i in range(sdc_trajectory.shape[0] - 1)
    ]
    sdc_track_length = sum(sdc_track_length)

    # Rule 1
    if sdc_track_length < 10:
        return False

    print("sdc_track_length: ", sdc_track_length)

    # Rule 2
    if valid_array.sum() < 50:
        return False

    return True

def _get_agent_summary(state_dict, id):
    track = state_dict["position"]
    valid_track = track[state_dict["valid"], :2]
    distance = float(sum(np.linalg.norm(valid_track[i] - valid_track[i + 1]) for i in range(valid_track.shape[0] - 1)))
    valid_length = int(sum(state_dict["valid"]))

    continuous_valid_length = 0
    for v in state_dict["valid"]:
        if v:
            continuous_valid_length += 1
        if continuous_valid_length > 0 and not v:
            break

    return {
        "object_id": str(id),
        "track_length": int(len(track)),
        "distance": float(distance),
        "valid_length": int(valid_length),
        "continuous_valid_length": int(continuous_valid_length)
    }


def parse_data(input, output_path, _selective=False):
    cnt = 0
    scenario = scenario_pb2.Scenario()
    file_list = os.listdir(input)

    metadata_recorder = {}

    for file in tqdm(file_list):
        file_path = os.path.join(input, file)
        if ("tfrecord" not in file_path) or (not os.path.isfile(file_path)):
            continue
        dataset = tf.data.TFRecordDataset(file_path, compression_type="")
        for j, data in enumerate(dataset.as_numpy_iterator()):
            scenario.ParseFromString(data)

            md_scenario = SD()

            md_scenario[SD.ID] = scenario.scenario_id

            md_scenario[SD.VERSION] = DATA_VERSION

            # Please note that SDC track index is not identical to sdc_id.
            # sdc_id is a unique indicator to a track, while sdc_track_index is only the index of the sdc track
            # in the tracks datastructure.

            track_length = len(scenario.dynamic_map_states)

            tracks, sdc_id = extract_tracks(scenario.tracks, scenario.sdc_track_index, track_length)

            valid = validate_sdc_track(tracks[sdc_id][SD.STATE])
            if not valid:
                continue

            track_length = list(tracks.values())[0]["state"]["position"].shape[0]

            md_scenario[SD.LENGTH] = track_length


            num_agent_types = len(set(v["type"] for v in tracks.values()))
            if _selective and num_agent_types < 3:
                print("Skip scenario {} because of lack of participant types {}.".format(j, num_agent_types))
                continue

            md_scenario[SD.TRACKS] = tracks

            dynamic_states = extract_dynamic_map_states(scenario.dynamic_map_states, track_length)
            if _selective and not dynamic_states:
                print("Skip scenario {} because of lack of traffic light.".format(j))
                continue
            md_scenario[SD.DYNAMIC_MAP_STATES] = dynamic_states

            try:
                md_scenario[SD.MAP_FEATURES] = extract_map_features(scenario.map_features)
            except AssertionError as e:
                print("FILE: ", file, j)
                raise e

            compute_width(md_scenario[SD.MAP_FEATURES])

            md_scenario[SD.METADATA] = {}
            md_scenario[SD.METADATA][SD.COORDINATE] = MetaDriveType.COORDINATE_WAYMO
            md_scenario[SD.METADATA][SD.TIMESTEP] = \
                np.asarray([ts for ts in scenario.timestamps_seconds], np.float32)
            md_scenario[SD.METADATA][SD.METADRIVE_PROCESSED] = False
            md_scenario[SD.METADATA][SD.METADRIVE_PROCESSED] = False
            md_scenario[SD.METADATA][SD.SDC_ID] = str(sdc_id)
            md_scenario[SD.METADATA]["dataset"] = "waymo"
            md_scenario[SD.METADATA]["scenario_id"] = scenario.scenario_id
            md_scenario[SD.METADATA]["source_file"] = str(file)
            md_scenario[SD.METADATA]["track_length"] = track_length

            # === Waymo specific data. Storing them here ===
            md_scenario[SD.METADATA]["current_time_index"] = scenario.current_time_index
            md_scenario[SD.METADATA]["sdc_track_index"] = scenario.sdc_track_index

            # obj id
            md_scenario[SD.METADATA]["objects_of_interest"] = [str(obj) for obj in scenario.objects_of_interest]

            track_index = [obj.track_index for obj in scenario.tracks_to_predict]
            track_id = [str(scenario.tracks[ind].id) for ind in track_index]
            track_difficulty = [obj.difficulty for obj in scenario.tracks_to_predict]
            track_obj_type = [tracks[id]["type"] for id in track_id]
            md_scenario[SD.METADATA]["tracks_to_predict"] = {
                id: {
                    "track_index": track_index[count],
                    "track_id": id,
                    "difficulty": track_difficulty[count],
                    "object_type": track_obj_type[count]
                } for count, id in enumerate(track_id)
            }

            export_file_name = "sd_{}_{}.pkl".format(file, scenario.scenario_id)

            summary_dict = {}
            summary_dict["sdc"] = _get_agent_summary(state_dict=md_scenario.get_sdc_track()["state"], id=sdc_id)
            for track_id, track in md_scenario[SD.TRACKS].items():
                summary_dict[track_id] = _get_agent_summary(state_dict=track["state"], id=track_id)
            md_scenario[SD.METADATA]["summary"] = summary_dict

            metadata_recorder[export_file_name] = copy.deepcopy(md_scenario[SD.METADATA])

            md_scenario = md_scenario.to_dict()

            SD.sanity_check(md_scenario, check_self_type=True)

            # TODO: FIXME: Some thing more to be added.

            p = os.path.join(output_path, export_file_name)
            with open(p, "wb") as f:
                pickle.dump(md_scenario, f)
            print("Scenario {} is saved at: {}".format(cnt, p))
            cnt += 1
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="The data folder storing raw tfrecord from Waymo dataset.")
    parser.add_argument(
        "--output", default="processed_data", type=str, help="The data folder storing raw tfrecord from Waymo dataset."
    )
    parser.add_argument("--selective", action="store_true", help="Whether select high-diversity valuable scenario.")
    args = parser.parse_args()

    scenario_data_path = args.input

    output_path: str = os.path.dirname(scenario_data_path)
    output_path = os.path.join(output_path, args.output)
    os.makedirs(output_path, exist_ok=True)

    raw_data_path = scenario_data_path

    # parse raw data from input path to output path,
    # there is 1000 raw data in google cloud, each of them produce about 500 pkl file
    parse_data(raw_data_path, output_path, _selective=args.selective)
    sys.exit()
    # file_path = AssetLoader.file_path("waymo", "processed", "0.pkl", return_raw_style=False)
    # data = read_waymo_data(file_path)
    # draw_waymo_map(data)
