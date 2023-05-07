import pickle

from metadrive.engine.asset_loader import AssetLoader
from metadrive.scenario.scenario_description import ScenarioDescription

import logging

logger = logging.getLogger(__name__)

logger.warning(
    "This converters will de deprecated. "
    "Use tools in ScenarioNet instead: https://github.com/metadriverse/ScenarioNet."
)

if __name__ == '__main__':

    with open("waymo120/0408_output_final/dataset_summary.pkl", "rb") as f:
        summary_dict = pickle.load(f)

    new_summary = {}
    for obj_id, summary in summary_dict.items():

        if summary["number_summary"]["dynamic_object_states"] == 0:
            continue

        if summary["object_summary"]["sdc"]["distance"] < 80 or \
                summary["object_summary"]["sdc"]["continuous_valid_length"] < 50:
            continue

        if len(summary["number_summary"]["object_types"]) < 3:
            continue

        if summary["number_summary"]["object"] < 80:
            continue

        new_summary[obj_id] = summary

        if len(new_summary) >= 3:
            break

    file_path = AssetLoader.file_path("waymo", ScenarioDescription.DATASET.SUMMARY_FILE, return_raw_style=False)
    with open(file_path, "wb") as f:
        pickle.dump(new_summary, f)

    print(new_summary.keys())
