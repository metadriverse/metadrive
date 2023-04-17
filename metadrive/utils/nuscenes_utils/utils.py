import numpy as np
from nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion

from metadrive.scenario import ScenarioDescription as SD
from metadrive.type import MetaDriveType

EGO = "ego"
ALL_TYPE = {"noise": 'noise',
            "human.pedestrian.adult": 'adult',
            "human.pedestrian.child": 'child',
            "human.pedestrian.wheelchair": 'wheelchair',
            "human.pedestrian.stroller": 'stroller',
            "human.pedestrian.personal_mobility": 'p.mobility',
            "human.pedestrian.police_officer": 'police',
            "human.pedestrian.construction_worker": 'worker',
            "animal": 'animal',
            "vehicle.car": 'car',
            "vehicle.motorcycle": 'motorcycle',
            "vehicle.bicycle": 'bicycle',
            "vehicle.bus.bendy": 'bus.bendy',
            "vehicle.bus.rigid": 'bus.rigid',
            "vehicle.truck": 'truck',
            "vehicle.construction": 'constr. veh',
            "vehicle.emergency.ambulance": 'ambulance',
            "vehicle.emergency.police": 'police car',
            "vehicle.trailer": 'trailer',
            "movable_object.barrier": 'barrier',
            "movable_object.trafficcone": 'trafficcone',
            "movable_object.pushable_pullable": 'push/pullable',
            "movable_object.debris": 'debris',
            "static_object.bicycle_rack": 'bicycle racks',
            "flat.driveable_surface": 'driveable',
            "flat.sidewalk": 'sidewalk',
            "flat.terrain": 'terrain',
            "flat.other": 'flat.other',
            "static.manmade": 'manmade',
            "static.vegetation": 'vegetation',
            "static.other": 'static.other',
            "vehicle.ego": "ego"
            }

NOISE_TYPE = {"noise": 'noise',
              "animal": 'animal',
              "static_object.bicycle_rack": 'bicycle racks',
              "movable_object.pushable_pullable": 'push/pullable',
              "movable_object.debris": 'debris',
              "static.manmade": 'manmade',
              "static.vegetation": 'vegetation',
              "static.other": 'static.other',
              }

HUMAN_TYPE = {"human.pedestrian.adult": 'adult',
              "human.pedestrian.child": 'child',
              "human.pedestrian.wheelchair": 'wheelchair',
              "human.pedestrian.stroller": 'stroller',
              "human.pedestrian.personal_mobility": 'p.mobility',
              "human.pedestrian.police_officer": 'police',
              "human.pedestrian.construction_worker": 'worker',
              }

BICYCLE_TYPE = {
    "vehicle.bicycle": 'bicycle',
    "vehicle.motorcycle": 'motorcycle',
}

VEHICLE_TYPE = {
    "vehicle.car": 'car',
    "vehicle.bus.bendy": 'bus.bendy',
    "vehicle.bus.rigid": 'bus.rigid',
    "vehicle.truck": 'truck',
    "vehicle.construction": 'constr. veh',
    "vehicle.emergency.ambulance": 'ambulance',
    "vehicle.emergency.police": 'police car',
    "vehicle.trailer": 'trailer',
    "vehicle.ego": "ego",
}

OBSTACLE_TYPE = {
    "movable_object.barrier": 'barrier',
    "movable_object.trafficcone": 'trafficcone',
}

TERRAIN_TYPE = {
    "flat.driveable_surface": 'driveable',
    "flat.sidewalk": 'sidewalk',
    "flat.terrain": 'terrain',
    "flat.other": 'flat.other'}


def get_metadrive_type(obj_type):
    meta_type = ALL_TYPE[obj_type]
    md_type = None
    if ALL_TYPE[obj_type] == "barrier":
        md_type = MetaDriveType.TRAFFIC_BARRIER,
    elif ALL_TYPE[obj_type] == "trafficcone":
        md_type = MetaDriveType.TRAFFIC_CONE
    elif obj_type in VEHICLE_TYPE:
        md_type = MetaDriveType.VEHICLE
    elif obj_type in HUMAN_TYPE:
        md_type = MetaDriveType.PEDESTRIAN
    elif obj_type in BICYCLE_TYPE:
        md_type = MetaDriveType.CYCLIST
    else:
        print("Can not map type: {} to any MetaDrive Type".format(obj_type))

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

    for frame_idx in range(episode_len):
        # Record all agents' states (position, velocity, ...)
        for id, state in frames[frame_idx].items():
            # Fill type
            md_type, meta_type = get_metadrive_type(state["type"])

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
    return tracks


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
    result[SD.VERSION] = nuscenes.version
    result[SD.LENGTH] = len(frames)
    result[SD.METADATA] = {}
    result[SD.METADATA][SD.METADRIVE_PROCESSED] = True
    result[SD.METADATA]["dataset"] = "nuscenes"
    result[SD.METADATA]["map"] = nuscenes.get("log", scene_info["log_token"])["location"]
    result[SD.METADATA]["date"] = nuscenes.get("log", scene_info["log_token"])["date_captured"]

    result[SD.METADATA]["scenario_id"] = scene_token
    result[SD.METADATA]["sample_rate"] = scenario_log_interval
    result[SD.METADATA][SD.TIMESTEP] = \
        np.asarray([scenario_log_interval * i for i in range(len(frames))], dtype=np.float32)
    result[SD.TRACKS] = get_tracks_from_frames(frames)
    result[SD.METADATA][SD.SDC_ID] = "ego"

    # nuscenes.get("log", scene_info["log_token"])["map_token"]

    return result
