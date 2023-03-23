import os
import pickle

from tqdm import tqdm

try:
    import tensorflow as tf
except ImportError:
    pass
from metadrive.utils.waymo_utils.protos import scenario_pb2
from metadrive.utils.waymo_utils.utils import extract_tracks, extract_dynamic, extract_map, compute_width
import sys


def parse_data(input, output_path):
    cnt = 0
    scenario = scenario_pb2.Scenario()
    file_list = os.listdir(input)
    for file in tqdm(file_list):
        file_path = os.path.join(input, file)
        if not 'tfrecord' in file_path:
            continue
        dataset = tf.data.TFRecordDataset(file_path, compression_type='')
        for j, data in enumerate(dataset.as_numpy_iterator()):
            scenario.ParseFromString(data)
            scene = dict()
            scene['id'] = scenario.scenario_id

            scene['version'] = 'Mar23'  # March of 2023

            scene['ts'] = [ts for ts in scenario.timestamps_seconds]

            scene['tracks'], sdc_id = extract_tracks(scenario.tracks, scenario.sdc_track_index)

            scene['sdc_track_index'] = sdc_id

            scene['dynamic_map_states'] = extract_dynamic(scenario.dynamic_map_states)

            scene['map_features'] = extract_map(scenario.map_features)

            compute_width(scene['map_features'])

            p = os.path.join(output_path, f'{cnt}.pkl')
            with open(p, 'wb') as f:
                pickle.dump(scene, f)
            cnt += 1
    return


if __name__ == "__main__":
    case_data_path = "/home/shady/Downloads/test"
    try:
        os.mkdir(case_data_path + "_processed")
    except:
        pass
    raw_data_path = case_data_path
    processed_data_path = case_data_path + "_processed"
    # parse raw data from input path to output path,
    # there is 1000 raw data in google cloud, each of them produce about 500 pkl file
    parse_data(raw_data_path, processed_data_path)
    sys.exit()
    # file_path = AssetLoader.file_path("waymo", "processed", "0.pkl", return_raw_style=False)
    # data = read_waymo_data(file_path)
    # draw_waymo_map(data)
