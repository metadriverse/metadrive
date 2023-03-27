import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from metadrive.utils.waymo_utils.waymo_type import LaneType, AgentType, AgentTypeClass
from metadrive.utils.waymo_utils.waymo_type import RoadLineType, RoadEdgeType, RoadLineTypeClass, RoadEdgeTypeClass

try:
    import tensorflow as tf
except ImportError:
    pass
try:
    from metadrive.utils.waymo_utils.protos import scenario_pb2
except ImportError:
    pass
import pickle
import numpy as np


def extract_poly(message):
    x = [i.x for i in message]
    y = [i.y for i in message]
    z = [i.z for i in message]
    coord = np.stack((x, y, z), axis=1).astype("float32")
    return coord


def extract_boundaries(fb):
    b = []
    # b = np.zeros([len(fb), 4], dtype="int64")
    for k in range(len(fb)):
        c = dict()
        c["lane_start_index"] = fb[k].lane_start_index
        c["lane_end_index"] = fb[k].lane_end_index
        c["boundary_type"] = RoadLineType[fb[k].boundary_type]
        c["boundary_feature_id"] = fb[k].boundary_feature_id
        b.append(c)

    return b


def extract_neighbors(fb):
    nbs = []
    for k in range(len(fb)):
        nb = dict()
        nb["feature_id"] = fb[k].feature_id
        nb["self_start_index"] = fb[k].self_start_index
        nb["self_end_index"] = fb[k].self_end_index
        nb["neighbor_start_index"] = fb[k].neighbor_start_index
        nb["neighbor_end_index"] = fb[k].neighbor_end_index
        nb["boundaries"] = extract_boundaries(fb[k].boundaries)
        nbs.append(nb)
    return nbs


def extract_center(f):
    center = dict()
    f = f.lane
    center["speed_limit_mph"] = f.speed_limit_mph

    center["type"] = LaneType[f.type]

    center["polyline"] = extract_poly(f.polyline)

    center["interpolating"] = f.interpolating

    center["entry_lanes"] = [x for x in f.entry_lanes]

    center["exit_lanes"] = [x for x in f.exit_lanes]

    center["left_boundaries"] = extract_boundaries(f.left_boundaries)

    center["right_boundaries"] = extract_boundaries(f.right_boundaries)

    center["left_neighbor"] = extract_neighbors(f.left_neighbors)

    center["right_neighbor"] = extract_neighbors(f.right_neighbors)

    return center


def extract_line(f):
    line = dict()
    f = f.road_line
    line["type"] = RoadLineType[f.type]
    line["polyline"] = extract_poly(f.polyline)
    return line


def extract_edge(f):
    edge = dict()
    f_ = f.road_edge
    edge["type"] = RoadEdgeType[f_.type]
    edge["polyline"] = extract_poly(f_.polyline)

    return edge


def extract_stop(f):
    stop = dict()
    f = f.stop_sign
    stop["type"] = "STOP_SIGN"
    stop["lane"] = [x for x in f.lane]
    stop["position"] = np.array([f.position.x, f.position.y, f.position.z], dtype="float32")
    return stop


def extract_crosswalk(f):
    cross_walk = dict()
    f = f.crosswalk
    cross_walk["type"] = "CROSS_WALK"
    cross_walk["polyline"] = extract_poly(f.polygon)
    return cross_walk


def extract_bump(f):
    speed_bump_data = dict()
    f = f.speed_bump
    speed_bump_data["type"] = "SPEED_BUMP"
    speed_bump_data["polyline"] = extract_poly(f.polygon)

    return speed_bump_data


def extract_tracks(tracks, sdc_idx):
    ret = dict()

    for obj in tracks:
        obj_state = dict()
        obj_state["type"] = AgentType[obj.object_type]

        obj_state["state"] = {}

        x = [state.center_x for state in obj.states]
        y = [state.center_y for state in obj.states]
        z = [state.center_z for state in obj.states]
        obj_state["state"]["position"] = np.stack([x, y, z], 1).astype("float32")

        l = [state.length for state in obj.states]
        w = [state.width for state in obj.states]
        h = [state.height for state in obj.states]
        obj_state["state"]["size"] = np.stack([l, w, h], 1).astype("float32")

        heading = [state.heading for state in obj.states]
        obj_state["state"]["heading"] = np.array(heading, dtype="float32")[:, np.newaxis]

        vx = [state.velocity_x for state in obj.states]
        vy = [state.velocity_y for state in obj.states]
        obj_state["state"]["velocity"] = np.stack([vx, vy], 1).astype("float32")

        valid = [state.valid for state in obj.states]
        obj_state["state"]["valid"] = np.array(valid)[:, np.newaxis]

        obj_state["metadata"] = dict(
            track_length=obj_state["state"]["position"].shape[0], type=AgentType[obj.object_type], object_id=obj.id
        )

        ret[obj.id] = obj_state

    return ret, tracks[sdc_idx].id


def extract_map(f):
    map_features = {}

    for i in range(len(f)):
        f_i = f[i]
        id = f_i.id
        if f_i.HasField("lane"):
            map_features[id] = extract_center(f_i)

        if f_i.HasField("road_line"):
            map_features[id] = extract_line(f_i)

        if f_i.HasField("road_edge"):
            map_features[id] = extract_edge(f_i)

        if f_i.HasField("stop_sign"):
            map_features[id] = extract_stop(f_i)

        if f_i.HasField("crosswalk"):
            map_features[id] = extract_crosswalk(f_i)

        if f_i.HasField("speed_bump"):
            map_features[id] = extract_bump(f_i)

    return map_features


def extract_dynamic(f):
    dynamics = {}

    # FIXME: TODO: This function is not finished yet.
    # for i in range(len(f)):
    #     f_i = f[i].lane_states
    #     tls_t = []
    #     for j in range(len(f_i)):
    #         f_i_j = f_i[j]
    #         tls_t_j = dict()
    #         tls_t_j["lane"] = f_i_j.lane
    #         tls_t_j["state"] = TrafficSignal[f_i_j.state]
    #         tls_t_j["stop_point"] = np.array(
    #             [f_i_j.stop_point.x, f_i_j.stop_point.y, f_i_j.stop_point.z], dtype="float32"
    #         )
    #         tls_t.append(tls_t_j)
    #
    #     dynamics.append(tls_t)

    return dynamics


class CustomUnpickler(pickle.Unpickler):
    def __init__(self, load_old_case, *args, **kwargs):
        raise DeprecationWarning("Now we don't pickle any customized data type, so this class is deprecated now")
        super(CustomUnpickler, self).__init__(*args, **kwargs)
        self.load_old_case = load_old_case

    def find_class(self, module, name):
        if self.load_old_case:
            if name == "AgentType":
                return AgentTypeClass
            elif name == "RoadLineType":
                return RoadLineTypeClass
            elif name == "RoadEdgeType":
                return RoadEdgeTypeClass
            return super().find_class(module, name)
        else:
            return super().find_class(module, name)


def read_waymo_data(file_path):
    with open(file_path, "rb") as f:
        # unpickler = CustomUnpickler(f)
        data = pickle.load(f)
    new_track = {}
    for key, value in data["tracks"].items():
        new_track[str(key)] = value
    data["tracks"] = new_track
    data["sdc_track_index"] = str(data["sdc_track_index"])
    return data


def draw_waymo_map(data):
    figure(figsize=(8, 6), dpi=500)
    for key, value in data["map_features"].items():
        if value.get("type", None) == "center_lane":
            plt.scatter([x[0] for x in value["polyline"]], [y[1] for y in value["polyline"]], s=0.5)
        elif value.get("type", None) == "road_edge":
            plt.scatter([x[0] for x in value["polyline"]], [y[1] for y in value["polyline"]], s=0.5, c=(0, 0, 0))
        # elif value.get("type", None) == "road_line":
        #     plt.scatter([x[0] for x in value["polyline"]], [y[1] for y in value["polyline"]], s=0.5, c=(0.8,0.8,0.8))
    plt.show()


# return the nearest point"s index of the line
def nearest_point(point, line):
    dist = np.square(line - point)
    dist = np.sqrt(dist[:, 0] + dist[:, 1])
    return np.argmin(dist)


def extract_width(map, polyline, boundary):
    l_width = np.zeros(polyline.shape[0], dtype="float32")
    for b in boundary:
        lb = map[b["boundary_feature_id"]]
        b_polyline = lb["polyline"][:, :2]

        start_p = polyline[b["lane_start_index"]]
        start_index = nearest_point(start_p, b_polyline)
        seg_len = b["lane_end_index"] - b["lane_start_index"]
        end_index = min(start_index + seg_len, lb["polyline"].shape[0] - 1)
        length = min(end_index - start_index, b["lane_end_index"] - b["lane_start_index"]) + 1
        self_range = range(b["lane_start_index"], b["lane_start_index"] + length)
        bound_range = range(start_index, start_index + length)
        centerLane = polyline[self_range]
        bound = b_polyline[bound_range]
        dist = np.square(centerLane - bound)
        dist = np.sqrt(dist[:, 0] + dist[:, 1])
        l_width[self_range] = dist
    return l_width


def compute_width(map):
    for id, lane in map.items():

        if not "LANE" in lane["type"]:
            continue

        width = np.zeros((lane["polyline"].shape[0], 2), dtype="float32")

        width[:, 0] = extract_width(map, lane["polyline"][:, :2], lane["left_boundaries"])
        width[:, 1] = extract_width(map, lane["polyline"][:, :2], lane["right_boundaries"])

        width[width[:, 0] == 0, 0] = width[width[:, 0] == 0, 1]
        width[width[:, 1] == 0, 1] = width[width[:, 1] == 0, 0]

        lane["width"] = width
    return


# parse raw data from input path to output path


def convert_polyline_to_metadrive(waymo_polyline):
    """
    Waymo lane is in a different coordinate system, using them after converting
    """
    return np.array([np.array([p[0], -p[1]]) for p in waymo_polyline])
