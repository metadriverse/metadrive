from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from metadrive.engine.asset_loader import AssetLoader
try:
    import tensorflow as tf
except ImportError:
    pass
from metadrive.utils.waymo_utils.protos import scenario_pb2
import os
import pickle
import numpy as np


def extract_poly(message):
    x = [i.x for i in message]
    y = [i.y for i in message]
    z = [i.z for i in message]
    coord = np.stack((x, y, z), axis=1)
    return coord


def extract_boundaries(fb):

    b = np.zeros([len(fb), 4], dtype='int64')
    for k in range(len(fb)):
        b[k, 0] = fb[k].lane_start_index
        b[k, 1] = fb[k].lane_end_index
        b[k, 2] = fb[k].boundary_feature_id
        b[k, 3] = fb[k].boundary_type
    return b


def extract_neighbors(fb):
    nb_list = []
    for k in range(len(fb)):
        nb = dict()
        nb['id'] = fb[k].feature_id
        nb['indexes'] = [
            fb[k].self_start_index, fb[k].self_end_index, fb[k].neighbor_start_index, fb[k].neighbor_end_index
        ]
        nb['boundaries'] = extract_boundaries(fb[k].boundaries)
        nb_list.append(nb)
    return nb_list


def extract_center(f):
    center = dict()
    f = f.lane

    center['type'] = 'center_lane'

    center['polyline'] = extract_poly(f.polyline)

    center['interpolating'] = f.interpolating

    center['entry'] = [x for x in f.entry_lanes]

    center['exit'] = [x for x in f.exit_lanes]

    center['left_boundaries'] = extract_boundaries(f.left_boundaries)

    center['right_boundaries'] = extract_boundaries(f.right_boundaries)

    center['left_neighbor'] = extract_neighbors(f.left_neighbors)

    center['right_neighbor'] = extract_neighbors(f.right_neighbors)

    return center


def extract_line(f):
    line = dict()
    f = f.road_line
    line['type'] = 'road_line'
    line['polyline'] = extract_poly(f.polyline)

    return line


def extract_edge(f):
    edge = dict()

    f = f.road_edge
    edge['type'] = 'road_edge'
    edge['polyline'] = extract_poly(f.polyline)

    return edge


def extract_stop(f):
    stop = dict()
    f = f.stop_sign
    stop['type'] = 'stop_sign'
    stop['lanes'] = [x for x in f.lane]
    stop['pos'] = [f.position.x, f.position.y, f.position.z]
    return stop


def extract_crosswalk(f):
    cross_walk = dict()
    f = f.crosswalk
    cross_walk['sign'] = 'cross_walk'
    cross_walk['polygon'] = extract_poly(f.polygon)
    return cross_walk


def extract_bump(f):
    speed_bump = dict()
    f = f.speed_bump
    speed_bump['type'] = speed_bump
    speed_bump['polygon'] = extract_poly(f.polygon)

    return speed_bump


def extract_tracks(f):
    track = dict()

    for i in range(len(f)):
        agent = dict()
        agent['type'] = f[i].object_type

        x = [state.center_x for state in f[i].states]
        y = [state.center_y for state in f[i].states]
        z = [state.center_z for state in f[i].states]
        l = [state.length for state in f[i].states]
        w = [state.width for state in f[i].states]
        h = [state.height for state in f[i].states]
        head = [state.heading for state in f[i].states]
        vx = [state.velocity_x for state in f[i].states]
        vy = [state.velocity_y for state in f[i].states]
        valid = [state.valid for state in f[i].states]
        agent['state'] = np.stack((x, y, z, l, w, h, head, vx, vy, valid), 1)
        track[f[i].id] = agent

    return track


def extract_map(f):
    data_dict = dict()

    for i in range(len(f)):
        if f[i].HasField('lane'):
            data_dict[f[i].id] = extract_center(f[i])

        if f[i].HasField('road_line'):
            data_dict[f[i].id] = extract_line(f[i])

        if f[i].HasField('road_edge'):
            data_dict[f[i].id] = extract_edge(f[i])

        if f[i].HasField('stop_sign'):
            data_dict[f[i].id] = extract_stop(f[i])

        if f[i].HasField('crosswalk'):
            data_dict[f[i].id] = extract_crosswalk(f[i])

        if f[i].HasField('speed_bump'):
            data_dict[f[i].id] = extract_bump(f[i])

    return data_dict


def extract_dynamic(f):
    dynamics = []
    for i in range(len(f)):
        states = f[i].lane_states
        one = dict()
        for j in range(len(states)):
            state = dict()
            state['state'] = states[j].state
            state['stop_point'] = [states[j].stop_point.x, states[j].stop_point.y, states[j].stop_point.z]
            one[states[j].lane] = state
    dynamics.append(one)

    return dynamics


def read_waymo_data(file_path):
    with open(file_path, "rb+") as waymo_file:
        data = pickle.load(waymo_file)
    return data


def draw_waymo_map(data):
    figure(figsize=(8, 6), dpi=500)
    for key, value in data["map"].items():
        if value.get("type", None) == "center_lane":
            plt.scatter([x[0] for x in value["polyline"]], [y[1] for y in value["polyline"]], s=0.5)
        elif value.get("type", None) == "road_edge":
            plt.scatter([x[0] for x in value["polyline"]], [y[1] for y in value["polyline"]], s=0.5, c=(0, 0, 0))
        # elif value.get("type", None) == "road_line":
        #     plt.scatter([x[0] for x in value["polyline"]], [y[1] for y in value["polyline"]], s=0.5, c=(0.8,0.8,0.8))
    plt.show()


# parse raw data from input path to output path
def parse_data(inut_path, output_path):
    cnt = 0
    scenario = scenario_pb2.Scenario()
    file_list = os.listdir(inut_path)
    for file in file_list:
        file_path = os.path.join(inut_path, file)
        if not 'scenario' in file_path:
            continue
        dataset = tf.data.TFRecordDataset(file_path, compression_type='')
        for j, data in enumerate(dataset.as_numpy_iterator()):
            scenario.ParseFromString(data)
            scene = dict()
            scene['id'] = scenario.scenario_id
            scene['ts'] = [ts for ts in scenario.timestamps_seconds]
            scene['tracks'] = extract_tracks(scenario.tracks)
            scene['dynamic_map_states'] = extract_dynamic(scenario.dynamic_map_states)
            scene['sdc_index'] = scenario.sdc_track_index
            # scene['interact_tracks'] = [x for x in scenario.objects_of_interest]
            # scene['motion_tracks'] = [x for x in scenario.tracks_to_predict]
            scene['map'] = extract_map(scenario.map_features)
            p = os.path.join(output_path, f'{cnt}.pkl')
            with open(p, 'wb') as f:
                pickle.dump(scene, f)
            cnt += 1
    return


if __name__ == "__main__":

    raw_data_path = AssetLoader.file_path("waymo","raw",  linux_style=False)
    processed_data_path = AssetLoader.file_path("waymo","processed", linux_style=False)
    # parse raw data from input path to output path,
    # there is 1000 raw data in google cloud, each of them produce about 500 pkl file
    parse_data(raw_data_path,processed_data_path)

    # file_path = AssetLoader.file_path("waymo", "test.pkl", linux_style=False)
    # data = read_waymo_data(file_path)
    # draw_waymo_map(data)
