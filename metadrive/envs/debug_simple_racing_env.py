import argparse
from typing import Union

from metadrive import MetaDriveEnv
from metadrive.component.map.pg_map import PGMap
from metadrive.component.pg_space import Parameter
from metadrive.component.pgblock.curve import CurveWithGuardrail
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.straight import StraightWithGuardrail
from metadrive.constants import HELP_MESSAGE
from metadrive.manager.pg_map_manager import PGMapManager


class RacingMap(PGMap):
    def __init__(self, map_config: dict = None, random_seed=None, map_type=None):
        super(RacingMap, self).__init__(map_config=map_config, random_seed=random_seed)
        self.map_type = map_type

    def _generate(self):
        # if self.map_type == "train":

        FirstPGBlock.ENTRANCE_LENGTH = 0.5
        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"

        init_block = FirstPGBlock(self.road_network, 10.0, 3, parent_node_path, physics_world, 1)
        self.blocks.append(init_block)

        curve_direction = 1
        curve_len = 50

        block_c1 = CurveWithGuardrail(1, init_block.get_socket(0), self.road_network, 1)
        block_c1.construct_from_config(
            {
                Parameter.length: curve_len,
                Parameter.radius: 50,
                Parameter.angle: 90,
                Parameter.dir: curve_direction,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c1)

        block_s2 = StraightWithGuardrail(2, block_c1.get_socket(0), self.road_network, 1, )
        block_s2.construct_from_config({
            Parameter.length: 5,
        }, parent_node_path, physics_world)
        self.blocks.append(block_s2)

        block_c2 = CurveWithGuardrail(3, block_s2.get_socket(0), self.road_network, 1)
        block_c2.construct_from_config(
            {
                Parameter.length: curve_len,
                Parameter.radius: 50,
                Parameter.angle: 90,
                Parameter.dir: curve_direction,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c2)

        block_c3 = CurveWithGuardrail(4, block_c2.get_socket(0), self.road_network, 1)
        block_c3.construct_from_config(
            {
                Parameter.length: curve_len,
                Parameter.radius: 50,
                Parameter.angle: 90,
                Parameter.dir: curve_direction,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c3)

        block_s3 = StraightWithGuardrail(5, block_c3.get_socket(0), self.road_network, 1)
        block_s3.construct_from_config({
            Parameter.length: 5,
        }, parent_node_path, physics_world)
        self.blocks.append(block_s3)

        block_c4 = CurveWithGuardrail(6, block_s3.get_socket(0), self.road_network, 1)
        block_c4.construct_from_config(
            {
                Parameter.length: curve_len,
                Parameter.radius: 50,
                Parameter.angle: 90,
                Parameter.dir: curve_direction,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c4)

        pos_from_node = list(self.road_network.graph.keys())[-1]
        pos_to_node = list(self.road_network.graph[pos_from_node].keys())[-1]
        self.road_network.graph[pos_to_node] = {'>': self.road_network.graph[pos_from_node][pos_to_node]}


class RacingMapManager(PGMapManager):
    def __init__(self, map_type=None):
        super(RacingMapManager, self).__init__()
        self.map_type = map_type

    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(
                RacingMap, map_config=config["map_config"], random_seed=None, map_type=self.map_type
            )
        else:
            assert len(self.spawned_objects) == 1, "It is supposed to contain one map in this manager"
            _map = self.spawned_objects.values()[0]
        self.load_map(_map)


class RacingEnv(MetaDriveEnv):
    def __init__(self, config: Union[dict, None] = None, map_type=None):
        super(RacingEnv, self).__init__(config)
        self.map_type = map_type

    def setup_engine(self):
        super(RacingEnv, self).setup_engine()
        map_manager = RacingMapManager()
        map_type = "train"
        map_manager.map_type = map_type
        self.engine.update_manager("map_manager", map_manager)

    def initial_setup_circular_tracks(self):
        self.vehicle.config["destination"] = list(self.current_map.road_network.graph.keys())[-2]
        self.vehicle.navigation.reset(self.vehicle)


if __name__ == "__main__":
    Racing_config = dict(
        # controller="joystick",
        # num_agents=2,
        use_render=False,
        manual_control=True,
        traffic_density=0.005,
        num_scenarios=1000,
        random_agent_model=False,
        debug=True,
        top_down_camera_initial_x=95,
        top_down_camera_initial_y=15,
        top_down_camera_initial_z=120,
        # random_lane_width=True,
        # random_lane_num=True,
        on_continuous_line_done=False,
        out_of_route_done=False,
        vehicle_config=dict(show_lidar=False, show_navi_mark=False),
    )

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    env = RacingEnv(Racing_config)
    changed_dest = False
    changed_time = 0
    racing_rounds = 3

    try:
        o, _ = env.reset()
        print(HELP_MESSAGE)

        complete_checkpoints = env.vehicle.navigation.checkpoints
        real_destination = list(env.current_map.road_network.graph.keys())[-1]
        env.vehicle.config["destination"] = list(env.current_map.road_network.graph.keys())[-2]
        env.vehicle.navigation.reset(env.vehicle)
        # env.initial_setup_circular_tracks()
        env.vehicle.expert_takeover = True
        g = 0

        for i in range(1, 1000000000000):
            # print(i)
            o, r, tm, tc, info = env.step([0, 0])
            g += r

            current_lane_index = env.vehicle.lane_index[1]
            lane, lane_index, on_lane = env.vehicle.navigation._get_current_lane(env.vehicle)

            if (len(complete_checkpoints) - 1 > complete_checkpoints.index(current_lane_index) >=
                len(complete_checkpoints) - 3) and not changed_dest:
                env.vehicle.config["destination"] = real_destination
                env.vehicle.navigation.reset(env.vehicle)
                changed_dest = True
                changed_time += 1

            # print(env.vehicle.lane_index)

            # obtained_checkpoints = env.vehicle.navigation.get_checkpoints()
            # checkpoints = env.vehicle.navigation.checkpoints
            # print(len(checkpoints), len(env.current_map.road_network.graph.keys()))
            # print(obtained_checkpoints)
            env.render(mode="topdown")
            # if info["arrive_dest"]:
            #     checkpoint = env.vehicle.navigation.checkpoints
            #     print(len(checkpoints))
            #     # env.vehicle.navigation
            #     pass
            if (tm or tc) and info["arrive_dest"]:
                if changed_time >= 3:
                    env.reset(env.current_seed + 1)
                    env.current_track_vehicle.expert_takeover = True
                    print("rewards: ", g)
                    exit()

                else:

                    env.vehicle.config["destination"] = list(env.current_map.road_network.graph.keys())[-2]
                    env.vehicle.navigation.reset(env.vehicle)
                    changed_dest = False

    except Exception as e:
        raise e
    finally:
        env.close()
