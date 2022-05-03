from enum import Enum
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from metadrive.engine.asset_loader import AssetLoader

try:
    import tensorflow as tf
except ImportError:
    pass
try:
    from metadrive.utils.waymo_utils.protos import scenario_pb2
except ImportError:
    pass
import os
import pickle
import numpy as np


class RoadLineType(Enum):
    UNKNOWN = 0
    BROKEN_SINGLE_WHITE = 1
    SOLID_SINGLE_WHITE = 2
    SOLID_DOUBLE_WHITE = 3
    BROKEN_SINGLE_YELLOW = 4
    BROKEN_DOUBLE_YELLOW = 5
    SOLID_SINGLE_YELLOW = 6
    SOLID_DOUBLE_YELLOW = 7
    PASSING_DOUBLE_YELLOW = 8

    @staticmethod
    def is_road_line(line):
        return True if line.__class__ == RoadLineType else False

    @staticmethod
    def is_yellow(line):
        return True if line in [
            RoadLineType.SOLID_DOUBLE_YELLOW, RoadLineType.PASSING_DOUBLE_YELLOW, RoadLineType.SOLID_SINGLE_YELLOW,
            RoadLineType.BROKEN_DOUBLE_YELLOW, RoadLineType.BROKEN_SINGLE_YELLOW
        ] else False

    @staticmethod
    def is_broken(line):
        return True if line in [
            RoadLineType.BROKEN_DOUBLE_YELLOW, RoadLineType.BROKEN_SINGLE_YELLOW, RoadLineType.BROKEN_SINGLE_WHITE
        ] else False


class RoadEdgeType(Enum):
    UNKNOWN = 0
    # Physical road boundary that doesn't have traffic on the other side (e.g., a curb or the k-rail on the right side of a freeway).
    BOUNDARY = 1
    # Physical road boundary that separates the car from other traffic (e.g. a k-rail or an island).
    MEDIAN = 2

    @staticmethod
    def is_road_edge(edge):
        return True if edge.__class__ == RoadEdgeType else False

    @staticmethod
    def is_sidewalk(edge):
        return True if edge == RoadEdgeType.BOUNDARY else False


class AgentType(Enum):
    UNSET = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHER = 4


def extract_poly(message):
    x = [i.x for i in message]
    y = [i.y for i in message]
    z = [i.z for i in message]
    coord = np.stack((x, y, z), axis=1)
    return coord


def extract_boundaries(fb):
    b = []
    # b = np.zeros([len(fb), 4], dtype='int64')
    for k in range(len(fb)):
        c = dict()
        c['index'] = [fb[k].lane_start_index, fb[k].lane_end_index]
        c['type'] = RoadLineType(fb[k].boundary_type)
        c['id'] = fb[k].boundary_feature_id
        b.append(c)

    return b


def extract_neighbors(fb):
    nbs = []
    for k in range(len(fb)):
        nb = dict()
        nb['id'] = fb[k].feature_id
        nb['indexes'] = [
            fb[k].self_start_index, fb[k].self_end_index, fb[k].neighbor_start_index, fb[k].neighbor_end_index
        ]
        nb['indexes'] = [
            fb[k].self_start_index, fb[k].self_end_index, fb[k].neighbor_start_index, fb[k].neighbor_end_index
        ]
        nb['boundaries'] = extract_boundaries(fb[k].boundaries)
        nb['id'] = fb[k].feature_id
        nbs.append(nb)
    return nbs


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
    line['type'] = RoadLineType(f.type)
    line['polyline'] = extract_poly(f.polyline)
    return line


def extract_edge(f):
    edge = dict()
    f = f.road_edge
    edge['type'] = RoadEdgeType(f.type)
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


def extract_tracks(f, sdc_idx):
    track = dict()

    for i in range(len(f)):
        agent = dict()
        agent['type'] = AgentType(f[i].object_type)
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

    return track, f[sdc_idx].id


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


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'AgentType':
            return AgentType
        elif name == "RoadLineType":
            return RoadLineType
        elif name == "RoadEdgeType":
            return RoadEdgeType
        return super().find_class(module, name)


def read_waymo_data(file_path):
    data = CustomUnpickler(open(file_path, "rb+")).load()
    new_track = {}
    for key, value in data["tracks"].items():
        new_track[str(key)] = value
    data["tracks"] = new_track
    data["sdc_index"] = str(data["sdc_index"])
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


# return the nearest point's index of the line
def nearest_point(point, line):
    dist = np.square(line - point)
    dist = np.sqrt(dist[:, 0] + dist[:, 1])
    return np.argmin(dist)


def extract_width(map, polyline, boundary):
    l_width = np.zeros(polyline.shape[0])
    for b in boundary:
        lb = map[b['id']]
        b_polyline = lb['polyline'][:, :2]

        start_p = polyline[b['index'][0]]
        start_index = nearest_point(start_p, b_polyline)
        seg_len = b['index'][1] - b['index'][0]
        end_index = min(start_index + seg_len, lb['polyline'].shape[0] - 1)
        leng = min(end_index - start_index, b['index'][1] - b['index'][0]) + 1
        self_range = range(b['index'][0], b['index'][0] + leng)
        bound_range = range(start_index, start_index + leng)
        centerLane = polyline[self_range]
        bound = b_polyline[bound_range]
        dist = np.square(centerLane - bound)
        dist = np.sqrt(dist[:, 0] + dist[:, 1])
        l_width[self_range] = dist
    return l_width


def compute_width(map):
    for key in map.keys():
        if not 'type' in map[key] or map[key]['type'] != 'center_lane':
            continue
        lane = map[key]

        width = np.zeros((lane['polyline'].shape[0], 2))

        width[:, 0] = extract_width(map, lane['polyline'][:, :2], lane['left_boundaries'])
        width[:, 1] = extract_width(map, lane['polyline'][:, :2], lane['right_boundaries'])

        width[width[:, 0] == 0, 0] = width[width[:, 0] == 0, 1]
        width[width[:, 1] == 0, 1] = width[width[:, 1] == 0, 0]

        lane['width'] = width
    return


# parse raw data from input path to output path
def parse_data(inut_path, output_path):
    cnt = 0
    scenario = scenario_pb2.Scenario()
    file_list = os.listdir(inut_path)
    for file in tqdm(file_list):
        file_path = os.path.join(inut_path, file)
        if not 'scenario' in file_path:
            continue
        dataset = tf.data.TFRecordDataset(file_path, compression_type='')
        for j, data in enumerate(dataset.as_numpy_iterator()):
            scenario.ParseFromString(data)
            scene = dict()
            scene['id'] = scenario.scenario_id
            scene['ts'] = [ts for ts in scenario.timestamps_seconds]
            scene['tracks'], sdc_id = extract_tracks(scenario.tracks, scenario.sdc_track_index)
            scene['dynamic_map_states'] = extract_dynamic(scenario.dynamic_map_states)
            scene['sdc_index'] = sdc_id
            # scene['interact_tracks'] = [x for x in scenario.objects_of_interest]
            # scene['motion_tracks'] = [x for x in scenario.tracks_to_predict]
            scene['map'] = extract_map(scenario.map_features)
            compute_width(scene['map'])
            p = os.path.join(output_path, f'{cnt}.pkl')
            with open(p, 'wb') as f:
                pickle.dump(scene, f)
            cnt += 1
    return


def convert_polyline_to_metadrive(waymo_polyline):
    """
    Waymo lane is in a different coordinate system, using them after converting
    """
    return [np.array([p[0], -p[1]]) for p in waymo_polyline]


if __name__ == "__main__":
    case_data_path = sys.argv[1]
    os.mkdir(case_data_path + "_processed")
    raw_data_path = case_data_path
    processed_data_path = case_data_path + "_processed"
    # parse raw data from input path to output path,
    # there is 1000 raw data in google cloud, each of them produce about 500 pkl file
    parse_data(raw_data_path, processed_data_path)

    # file_path = AssetLoader.file_path("waymo", "processed", "0.pkl", return_raw_style=False)
    # data = read_waymo_data(file_path)
    # draw_waymo_map(data)
