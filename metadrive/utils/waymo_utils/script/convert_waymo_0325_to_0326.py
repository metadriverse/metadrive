"""
This script is introduced temporarily for PR #375: https://github.com/metadriverse/metadrive/pull/375
"""
import pickle
import sys

from metadrive.engine.asset_loader import AssetLoader


def convert_case(file_path, new_path):
    with open(file_path, "rb+") as file:
        data = pickle.load(file)
    for obj_id, obj_dict in data["tracks"].items():
        new_obj_dict = dict(
            type=obj_dict["type"],
            state={k: v
                   for k, v in obj_dict.items() if k != "type"},
            metadata=dict(track_length=obj_dict["position"].shape[0], type=obj_dict["type"], object_id=obj_id)
        )
        data["tracks"][obj_id] = new_obj_dict
    with open(new_path, "wb+") as file:
        pickle.dump(data, file)


if __name__ == "__main__":
    for i in range(3):
        file_path = AssetLoader.file_path("waymo", "{}.pkl".format(i), return_raw_style=False)
        new_file_path = AssetLoader.file_path("waymo", "{}.pkl".format(i), return_raw_style=False)
        convert_case(file_path, new_file_path)
    sys.exit()
