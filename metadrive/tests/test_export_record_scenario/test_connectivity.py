import os
import copy
import os
import pickle
import shutil

from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.replay_policy import ReplayEgoCarPolicy


def test_search_path(render_export_env=False, render_load_env=False):
    # Origin Data
    env = MetaDriveEnv(
        dict(
            start_seed=0,
            use_render=render_export_env,
            num_scenarios=1,
            map_config={
                BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                BaseMap.GENERATE_CONFIG: "OSXCTrCS",  # it can be a file path / block num / block ID sequence
                BaseMap.LANE_WIDTH: 3.5,
                BaseMap.LANE_NUM: 1,
                "exit_length": 50,
            },
            agent_policy=IDMPolicy
        )
    )
    policy = lambda x: [0, 1]
    dir = None

    try:
        scenarios, done_info = env.export_scenarios(policy, scenario_index=[i for i in range(1)])
        dir = os.path.join(os.path.dirname(__file__), "../test_component/test_export")
        os.makedirs(dir, exist_ok=True)
        for i, data in scenarios.items():
            with open(os.path.join(dir, "{}.pkl".format(i)), "wb+") as file:
                pickle.dump(data, file)
        node_roadnet = copy.deepcopy(env.current_map.road_network)
        env.close()

        # Loaded Data
        env = ScenarioEnv(
            dict(agent_policy=ReplayEgoCarPolicy, data_directory=dir, use_render=render_load_env, num_scenarios=1)
        )
        scenarios, done_info = env.export_scenarios(policy, scenario_index=[i for i in range(1)])
        dir = os.path.join(os.path.dirname(__file__), "../test_component/test_export")
        os.makedirs(dir, exist_ok=True)
        for i, data in scenarios.items():
            with open(os.path.join(dir, "{}.pkl".format(i)), "wb+") as file:
                pickle.dump(data, file)
        env.close()

        # reload
        env = ScenarioEnv(
            dict(agent_policy=ReplayEgoCarPolicy, data_directory=dir, use_render=render_load_env, num_scenarios=1)
        )
        for index in range(1):
            env.reset(seed=index)
            done = False
            while not done:
                o, r, tm, tc, i = env.step([0, 0])
                done = tm or tc
        edge_roadnet = copy.deepcopy(env.current_map.road_network)
        all_node_lanes = node_roadnet.get_all_lanes()
        all_edge_lanes = edge_roadnet.get_all_lanes()
        diff = set(["{}".format(l.index) for l in all_node_lanes]) - set(["{}".format(l.index) for l in all_edge_lanes])
        assert len(diff) == 0
        nodes = node_roadnet.shortest_path('>', "8S0_0_")
        print(nodes)
        edges = edge_roadnet.shortest_path("('>', '>>', 0)", "('7C0_1_', '8S0_0_', 0)")

        def process_data(input_list):
            # Initialize the output list
            output_list = []

            for item in input_list:
                # Remove the outer double quotes and then split the string based on commas
                elements = item.strip('""').split(',')

                # Extract the first two elements, strip the unnecessary characters and append to the output list
                for elem in elements[:2]:
                    output_list.append(elem.strip(' "\'()'))

            # Remove duplicates while maintaining order
            output_list = list(dict.fromkeys(output_list))

            return output_list

        to_node = process_data(edges)
        print(to_node)
        assert to_node == nodes

    finally:
        env.close()
        if dir is not None:
            shutil.rmtree(dir)


if __name__ == '__main__':
    test_search_path()
