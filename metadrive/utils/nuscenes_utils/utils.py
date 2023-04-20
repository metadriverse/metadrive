import numpy as np
from nuscenes.map_expansion.arcline_path_utils import discretize_lane
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion

from metadrive.scenario import ScenarioDescription as SD
from metadrive.type import MetaDriveType
from metadrive.utils.nuscenes_utils.detection_type import ALL_TYPE, HUMAN_TYPE, BICYCLE_TYPE, VEHICLE_TYPE

EGO = "ego"


def get_metadrive_type(obj_type):
    meta_type = obj_type
    md_type = None
    if ALL_TYPE[obj_type] == "barrier":
        md_type = MetaDriveType.TRAFFIC_BARRIER
    elif ALL_TYPE[obj_type] == "trafficcone":
        md_type = MetaDriveType.TRAFFIC_CONE
    elif obj_type in VEHICLE_TYPE:
        md_type = MetaDriveType.VEHICLE
    elif obj_type in HUMAN_TYPE:
        md_type = MetaDriveType.PEDESTRIAN
    elif obj_type in BICYCLE_TYPE:
        md_type = MetaDriveType.CYCLIST

    # assert meta_type != MetaDriveType.UNSET and meta_type != "noise"
    return md_type, meta_type


def parse_frame(frame, nuscenes: NuScenes):
    ret = {}
    for obj_id in frame["anns"]:
        obj = nuscenes.get("sample_annotation", obj_id)
        ret[obj["instance_token"]] = {"position": obj["translation"],
                                      "obj_id": obj["instance_token"],
                                      "heading": quaternion_yaw(Quaternion(*obj["rotation"])),
                                      "rotation": obj["rotation"],
                                      "size": obj["size"],
                                      "visible": obj["visibility_token"],
                                      "attribute": [nuscenes.get("attribute", i)["name"] for i in
                                                    obj["attribute_tokens"]],
                                      "type": obj["category_name"]}
    ego_state = nuscenes.get("ego_pose", nuscenes.get("sample_data", frame["data"]["LIDAR_TOP"])["ego_pose_token"])

    ret[EGO] = {"position": ego_state["translation"],
                "obj_id": EGO,
                "heading": quaternion_yaw(Quaternion(*ego_state["rotation"])),
                "rotation": ego_state["rotation"],
                "type": "vehicle.car",
                # size https://en.wikipedia.org/wiki/Renault_Zoe
                "size": [4.08, 1.73, 1.56],
                }
    return ret


def get_tracks_from_frames(frames):
    episode_len = len(frames)
    # Fill tracks
    all_objs = set()
    for frame in frames:
        all_objs.update(frame.keys())
    tracks = {
        k: dict(
            type=MetaDriveType.UNSET,
            state=dict(
                position=np.zeros(shape=(episode_len, 3)),
                heading=np.zeros(shape=(episode_len,)),
                velocity=np.zeros(shape=(episode_len, 2)),
                valid=np.zeros(shape=(episode_len,)),
                length=np.zeros(shape=(episode_len, 1)),
                width=np.zeros(shape=(episode_len, 1)),
                height=np.zeros(shape=(episode_len, 1))
            ),
            metadata=dict(track_length=episode_len, type=MetaDriveType.UNSET, object_id=k, original_id=k)
        )
        for k in list(all_objs)
    }

    tracks_to_remove = set()

    for frame_idx in range(episode_len):
        # Record all agents' states (position, velocity, ...)
        for id, state in frames[frame_idx].items():
            # Fill type
            md_type, meta_type = get_metadrive_type(state["type"])
            tracks[id]["type"] = md_type
            tracks[id][SD.METADATA]["type"] = meta_type
            if md_type is None or md_type == MetaDriveType.UNSET:
                tracks_to_remove.add(id)
                continue

            tracks[id]["type"] = md_type
            tracks[id][SD.METADATA]["type"] = meta_type

            # Introducing the state item
            tracks[id]["state"]["position"][frame_idx] = state["position"]
            tracks[id]["state"]["heading"][frame_idx] = state["heading"]
            # tracks[id]["state"]["velocity"][frame_idx] = state["velocity"]
            tracks[id]["state"]["valid"][frame_idx] = 1

            tracks[id]["state"]["length"][frame_idx] = state["size"][0]
            tracks[id]["state"]["width"][frame_idx] = state["size"][1]
            tracks[id]["state"]["height"][frame_idx] = state["size"][2]

            tracks[id]["metadata"]["original_id"] = id
            tracks[id]["metadata"]["object_id"] = id
    for track in tracks_to_remove:
        track_data = tracks.pop(track)
        obj_type = track_data[SD.METADATA]["type"]
        print("Can not map type: {} to any MetaDrive Type".format(obj_type))
    return tracks


def get_map_features(scene_info, nuscenes: NuScenes, map_center, radius=250, sampling_rate=2):
    """
    Extract map features from nuscenes data. The objects in specified region will be returned. Sampling rate determines
    the distance between 2 points when extracting lane center line.
    """
    ret = {}
    map_name = nuscenes.get("log", scene_info["log_token"])["location"]
    map_api = NuScenesMap(dataroot=nuscenes.dataroot, map_name=map_name)

    layer_names = [
        # "line",
        # "polygon",
        # "node",
        # 'drivable_area',
        # 'road_segment',
        # 'road_block',
        'lane',
        # 'ped_crossing',
        # 'walkway',
        # 'stop_line',
        # 'carpark_area',
        'lane_connector',
        'road_divider',
        'lane_divider',
        'traffic_light'
    ]
    map_objs = map_api.get_records_in_radius(map_center[0], map_center[1], radius, layer_names)

    for id in map_objs["lane_divider"]:
        line_info = map_api.get("lane_divider", id)
        assert line_info["token"] == id
        line = map_api.extract_line(line_info["line_token"]).coords.xy
        line = [[line[0][i], line[1][i]] for i in range(len(line[0]))]
        ret[id] = {SD.TYPE: MetaDriveType.LINE_BROKEN_SINGLE_WHITE,
                   SD.POLYLINE: line}
    for id in map_objs["lane"]:
        lane_info = map_api.get("lane", id)
        assert lane_info["token"] == id
        boundary = map_api.extract_polygon(lane_info["polygon_token"]).boundary.xy
        boundary_polygon = [[boundary[0][i], boundary[1][i], 0.1] for i in range(len(boundary[0]))]
        boundary_polygon += [[boundary[0][i], boundary[1][i], 0.] for i in range(len(boundary[0]))]
        ret[id] = {SD.TYPE: MetaDriveType.LANE_SURFACE_STREET,
                   SD.POLYLINE: discretize_lane(map_api.arcline_path_3[id], resolution_meters=sampling_rate),
                   SD.POLYGON: boundary_polygon,
                   # TODO Add speed limit if needed
                   "speed_limit_kmh": 100}

    for id in map_objs["lane_connector"]:
        lane_info = map_api.get("lane_connector", id)
        assert lane_info["token"] == id
        boundary = map_api.extract_polygon(lane_info["polygon_token"]).boundary.xy
        boundary_polygon = [[boundary[0][i], boundary[1][i], 0.1] for i in range(len(boundary[0]))]
        boundary_polygon += [[boundary[0][i], boundary[1][i], 0.] for i in range(len(boundary[0]))]
        ret[id] = {SD.TYPE: MetaDriveType.LANE_SURFACE_STREET,
                   SD.POLYLINE: discretize_lane(map_api.arcline_path_3[id], resolution_meters=sampling_rate),
                   SD.POLYGON: boundary_polygon,
                   # TODO Add speed limit if needed
                   "speed_limit_kmh": 100}

    return ret


def convert_one_scene(scene_token: str, nuscenes: NuScenes, scenario_log_interval=0.5):
    scene_info = nuscenes.get("scene", scene_token)
    frames = []
    current_frame = nuscenes.get("sample", scene_info["first_sample_token"])
    while current_frame["token"] != scene_info["last_sample_token"]:
        frames.append(parse_frame(current_frame, nuscenes))
        current_frame = nuscenes.get("sample", current_frame["next"])
    frames.append(parse_frame(current_frame, nuscenes))
    assert current_frame["next"] == ""
    assert len(frames) == scene_info["nbr_samples"], "Number of sample mismatches! "

    result = SD()
    result[SD.ID] = scene_token
    result[SD.VERSION] = "nuscenes"+ nuscenes.version
    result[SD.LENGTH] = len(frames)
    result[SD.METADATA] = {}
    result[SD.METADATA][SD.METADRIVE_PROCESSED] = True
    result[SD.METADATA]["dataset"] = "nuscenes"
    result[SD.METADATA]["map"] = nuscenes.get("log", scene_info["log_token"])["location"]
    result[SD.METADATA]["date"] = nuscenes.get("log", scene_info["log_token"])["date_captured"]
    result[SD.METADATA]["coordinate"] = "right-handed"
    result[SD.METADATA]["scenario_id"] = scene_token
    result[SD.METADATA]["sample_rate"] = scenario_log_interval
    result[SD.METADATA][SD.TIMESTEP] = \
        np.asarray([scenario_log_interval * i for i in range(len(frames))], dtype=np.float32)
    result[SD.TRACKS] = get_tracks_from_frames(frames)
    result[SD.METADATA][SD.SDC_ID] = "ego"

    # TODO Traffic Light
    result[SD.DYNAMIC_MAP_STATES] = {}

    # map
    map_center = result[SD.TRACKS]["ego"]["state"]["position"][0]
    result[SD.MAP_FEATURES] = get_map_features(scene_info, nuscenes, map_center, 250)

    return result
