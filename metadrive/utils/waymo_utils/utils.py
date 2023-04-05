import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from metadrive.type import MetaDriveType
from metadrive.utils.math_utils import mph_to_kmh
from metadrive.utils.waymo_utils.waymo_type import WaymoLaneType, WaymoAgentType
from metadrive.utils.waymo_utils.waymo_type import WaymoRoadLineType, WaymoRoadEdgeType

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
from metadrive.scenario.scenario_description import ScenarioDescription


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
        c["boundary_type"] = WaymoRoadLineType.from_waymo(fb[k].boundary_type)
        c["boundary_feature_id"] = fb[k].boundary_feature_id
        for key in c:
            c[key] = str(c[key])
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
        for key in nb:
            nb[key] = str(nb[key])
        nb["boundaries"] = extract_boundaries(fb[k].boundaries)
        nbs.append(nb)
    return nbs


def extract_center(f):
    center = dict()
    f = f.lane
    center["speed_limit_mph"] = f.speed_limit_mph

    center["speed_limit_kmh"] = mph_to_kmh(f.speed_limit_mph)

    center["type"] = WaymoLaneType.from_waymo(f.type)

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
    line["type"] = WaymoRoadLineType.from_waymo(f.type)
    line["polyline"] = extract_poly(f.polyline)
    return line


def extract_edge(f):
    edge = dict()
    f_ = f.road_edge

    # TODO: Need to transform this to MetaDrive version
    edge["type"] = WaymoRoadEdgeType.from_waymo(f_.type)

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
        object_id = str(obj.id)

        obj_state = dict()

        waymo_string = WaymoAgentType.from_waymo(obj.object_type)  # Load waymo type string
        metadrive_type = MetaDriveType.from_waymo(waymo_string)  # Transform it to Waymo type string
        obj_state["type"] = metadrive_type

        obj_state["state"] = {}

        x = [state.center_x for state in obj.states]
        y = [state.center_y for state in obj.states]
        z = [state.center_z for state in obj.states]
        obj_state["state"]["position"] = np.stack([x, y, z], 1).astype("float32")

        l = [state.length for state in obj.states]
        w = [state.width for state in obj.states]
        h = [state.height for state in obj.states]
        # obj_state["state"]["size"] = np.stack([l, w, h], 1).astype("float32")
        obj_state["state"]["length"] = np.asarray(l).reshape(-1, 1)
        obj_state["state"]["width"] = np.asarray(w).reshape(-1, 1)
        obj_state["state"]["height"] = np.asarray(h).reshape(-1, 1)

        heading = [state.heading for state in obj.states]
        obj_state["state"]["heading"] = np.array(heading, dtype="float32")

        vx = [state.velocity_x for state in obj.states]
        vy = [state.velocity_y for state in obj.states]
        obj_state["state"]["velocity"] = np.stack([vx, vy], 1).astype("float32")

        valid = [state.valid for state in obj.states]
        obj_state["state"]["valid"] = np.array(valid, dtype=bool)

        obj_state["metadata"] = dict(
            track_length=obj_state["state"]["position"].shape[0], type=metadrive_type, object_id=object_id
        )

        ret[object_id] = obj_state

    return ret, str(tracks[sdc_idx].id)


def extract_map_features(map_features):
    ret = {}

    for lane_state in map_features:
        lane_id = str(lane_state.id)

        if lane_state.HasField("lane"):
            ret[lane_id] = extract_center(lane_state)

        if lane_state.HasField("road_line"):
            ret[lane_id] = extract_line(lane_state)

        if lane_state.HasField("road_edge"):
            ret[lane_id] = extract_edge(lane_state)

        if lane_state.HasField("stop_sign"):
            ret[lane_id] = extract_stop(lane_state)

        if lane_state.HasField("crosswalk"):
            ret[lane_id] = extract_crosswalk(lane_state)

        if lane_state.HasField("speed_bump"):
            ret[lane_id] = extract_bump(lane_state)

    return ret


def extract_dynamic_map_states(dynamic_map_states):
    processed_dynamics_map_states = {}
    track_length = len(dynamic_map_states)

    def _traffic_light_state_template(object_id):
        return dict(
            type=MetaDriveType.TRAFFIC_LIGHT,
            state=dict(
                stop_point=np.zeros([track_length, 3], dtype=np.float32),
                object_state=np.zeros([
                    track_length,
                ], dtype=int),
                lane=np.zeros([
                    track_length,
                ], dtype=int),
            ),
            metadata=dict(
                track_length=track_length, type=MetaDriveType.TRAFFIC_LIGHT, object_id=object_id, dataset="waymo"
            )
        )

    # FIXME: TODO: This function is not finished yet.
    for step_count, step_states in enumerate(dynamic_map_states):
        # Each step_states is the state of all objects in one time step
        lane_states = step_states.lane_states

        for object_state in lane_states:
            lane = object_state.lane

            # We will use lane index to serve as the traffic light index.
            if lane not in processed_dynamics_map_states:
                object_id = str(lane)  # Always use string to specify object id
                processed_dynamics_map_states[object_id] = _traffic_light_state_template(object_id=object_id)

            processed_dynamics_map_states[object_id]["state"]["lane"][step_count] = lane
            processed_dynamics_map_states[object_id]["state"]["object_state"][step_count] = object_state.state
            processed_dynamics_map_states[object_id]["state"]["stop_point"][step_count][0] = object_state.stop_point.x
            processed_dynamics_map_states[object_id]["state"]["stop_point"][step_count][1] = object_state.stop_point.y
            processed_dynamics_map_states[object_id]["state"]["stop_point"][step_count][2] = object_state.stop_point.z

    return processed_dynamics_map_states


class CustomUnpickler(pickle.Unpickler):
    def __init__(self, load_old_scenario, *args, **kwargs):
        raise DeprecationWarning("Now we don't pickle any customized data type, so this class is deprecated now")
        super(CustomUnpickler, self).__init__(*args, **kwargs)
        self.load_old_scenario = load_old_scenario

    def find_class(self, module, name):
        if self.load_old_scenario:
            raise ValueError("Old scenario is completely deprecated. Can't load it any more.")
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
    """
    TODO: This function transform data again. We should remove it and let MetaDrive read native data completely.
    """
    with open(file_path, "rb") as f:
        # unpickler = CustomUnpickler(f)
        data = pickle.load(f)
    data = ScenarioDescription(data)
    return data


def draw_waymo_map(data):
    figure(figsize=(8, 6), dpi=500)
    for key, value in data[ScenarioDescription.MAP_FEATURES].items():
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
        boundary_int = {k: int(v) if k != "boundary_type" else v for k, v in b.items()}  # All values are int

        b_feat_id = str(boundary_int["boundary_feature_id"])
        lb = map[b_feat_id]
        b_polyline = lb["polyline"][:, :2]

        start_p = polyline[boundary_int["lane_start_index"]]
        start_index = nearest_point(start_p, b_polyline)
        seg_len = boundary_int["lane_end_index"] - boundary_int["lane_start_index"]
        end_index = min(start_index + seg_len, lb["polyline"].shape[0] - 1)
        length = min(end_index - start_index, seg_len) + 1
        self_range = range(boundary_int["lane_start_index"], boundary_int["lane_start_index"] + length)
        bound_range = range(start_index, start_index + length)
        centerLane = polyline[self_range]
        bound = b_polyline[bound_range]
        dist = np.square(centerLane - bound)
        dist = np.sqrt(dist[:, 0] + dist[:, 1])
        l_width[self_range] = dist
    return l_width


def compute_width(map):
    for map_feat_id, lane in map.items():

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


def convert_polyline_to_metadrive(waymo_polyline, coordinate_transform=True):
    """
    Waymo lane is in a different coordinate system, using them after converting
    """
    waymo_polyline = np.asarray(waymo_polyline)
    if coordinate_transform:
        return np.stack([waymo_polyline[:, 0], -waymo_polyline[:, 1]], axis=1)
    else:
        return waymo_polyline
