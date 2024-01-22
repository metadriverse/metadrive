import copy
import os
import pickle

from metadrive.engine.asset_loader import AssetLoader
from metadrive.scenario import utils as sd_utils

if __name__ == '__main__':
    waymo_data = AssetLoader.file_path(
        AssetLoader.asset_path, "waymo", unix_style=False
    )  # Use the built-in datasets with simulator
    os.listdir(waymo_data)  # there are 3 waymo scenario files with one 'dataset_summary.pkl' in this example dataset.

    with open(waymo_data + '/dataset_summary.pkl', 'rb') as f:
        dataset_summary = pickle.load(f)

    # Get the dataset path
    dataset_path = waymo_data
    print("Dataset path: ", dataset_path)

    # Get the scenario .pkl file name
    _, scenario_ids, dataset_mapping = sd_utils.read_dataset_summary(dataset_path)

    for scenario_pkl_file in scenario_ids:

        # Get the relative path to the .pkl file
        print("The pkl file relative path: ", dataset_mapping[scenario_pkl_file])  # A empty path

        # Get the absolute path to the .pkl file
        abs_path_to_pkl_file = os.path.join(dataset_path, dataset_mapping[scenario_pkl_file], scenario_pkl_file)
        print("The pkl file absolute path: ", abs_path_to_pkl_file)

        # Call utility function in MD and get the ScenarioDescrption object
        scenario = sd_utils.read_scenario_data(abs_path_to_pkl_file)

        print("Before update: ", scenario['metadata']['number_summary'])

        SD = sd_utils.ScenarioDescription
        if hasattr(SD, "update_summaries"):
            SD.update_summaries(scenario)
        else:
            raise ValueError("Please update MetaDrive to latest version.")

        sd_utils.ScenarioDescription.update_summaries(scenario)

        print("After update: ", scenario['metadata']['number_summary'])

        # sanity check
        sd_scenario = scenario.to_dict()
        SD.sanity_check(sd_scenario, check_self_type=True)

        # dump
        with open(abs_path_to_pkl_file, "wb") as f:
            pickle.dump(sd_scenario, f)

        # Update summary
        print('\n\n Before', dataset_summary[scenario_pkl_file]['number_summary'])
        dataset_summary[scenario_pkl_file] = copy.deepcopy(sd_scenario[SD.METADATA])

        print('\n\n After', dataset_summary[scenario_pkl_file]['number_summary'])

    with open(waymo_data + '/dataset_summary.pkl', 'wb') as f:
        pickle.dump(dataset_summary, f)
