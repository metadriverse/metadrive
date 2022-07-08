from panda3d.core import NodePath

from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.create_pg_block_utils import CreateRoadFrom, ExtendStraightLane, get_lanes_on_road
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.road_network import Road
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.constants import LineType
from metadrive.engine.core.physics_world import PhysicsWorld
from metadrive.envs.marl_envs.marl_bottleneck import MultiAgentBottleneckEnv
from metadrive.manager.map_manager import MapManager
from metadrive.utils import Config, get_np_random
from metadrive.utils.space import ParameterSpace


class SecondPGBlock(FirstPGBlock):
    """
    A special Set, only used to create the Second block. One scene has only one Second block!!!
    """
    NODE_1 = ">"
    NODE_2 = ">>"
    NODE_3 = ">>>"
    NODE_4 = "1y0_0_"
    NODE_5 = "1y0_1_"
    PARAMETER_SPACE = ParameterSpace({})
    ID = "I"
    SOCKET_NUM = 1
    ENTRANCE_LENGTH = 10
    BOT_LENGTH = 15
    END_LENGTH = 60

    def __init__(
            self,
            global_network: NodeRoadNetwork,
            lane_width: float,
            lane_num: int,
            render_root_np: NodePath,
            physics_world: PhysicsWorld,
            length: float = 30,
            ignore_intersection_checking=False
    ):
        # place_holder = PGBlockSocket(Road(Decoration.start, Decoration.end), Road(Decoration.start, Decoration.end))
        super(SecondPGBlock, self).__init__(global_network=global_network, lane_width=lane_width, lane_num=lane_num,
                                            render_root_np=render_root_np, physics_world=physics_world, length=length,
                                            ignore_intersection_checking=ignore_intersection_checking)
        # assert length > self.ENTRANCE_LENGTH, (length, self.ENTRANCE_LENGTH)
        self._block_objects = []
        basic_lane = StraightLane(
            [0, lane_width * (lane_num - 1)], [self.ENTRANCE_LENGTH, lane_width * (lane_num - 1)],
            line_types=(LineType.BROKEN, LineType.SIDE),
            width=lane_width
        )


        # Node 1 -> Node 2
        ego_v_spawn_road = Road(self.NODE_1, self.NODE_2)
        # ego_v_spawn_road2 = Road(self.NODE_2, self.NODE_1)
        # print("this is second block")
        CreateRoadFrom(
            basic_lane,
            lane_num,
            ego_v_spawn_road,
            self.block_network,
            self._global_network,
            center_line_type=LineType.NONE,
            inner_lane_line_type=LineType.NONE,
            # side_lane_line_type=LineType.NONE,
            ignore_intersection_checking=self.ignore_intersection_checking
        )

        # Node 2 -> Node 1
        adverse_road = Road(self.NODE_2, self.NODE_1)
        lanes = get_lanes_on_road(ego_v_spawn_road, self.block_network)
        reference_lane = lanes[-1]
        num = len(lanes)
        width = reference_lane.width_at(0)
        assert isinstance(reference_lane, StraightLane)
        start_point = reference_lane.position(lanes[-1].length, -(num - 1) * width)
        end_point = reference_lane.position(0, -(num - 1) * width)

        symmetric_lane = StraightLane(
            start_point, end_point, width, lanes[-1].line_types, reference_lane.forbidden,
            reference_lane.speed_limit,
            reference_lane.priority
        )

        CreateRoadFrom(
            symmetric_lane,
            num,
            adverse_road,
            self.block_network,
            self._global_network,
            # ignore_start=ignore_start,
            # ignore_end=ignore_end,
            # side_lane_line_type=side_lane_line_type,
            inner_lane_line_type=LineType.NONE,
            center_line_type=LineType.NONE,
            ignore_intersection_checking=self.ignore_intersection_checking
        )

        # ----------------------------------------------------------------------------------------------------------------------------------
        next_lane = ExtendStraightLane(basic_lane, length - self.ENTRANCE_LENGTH, [LineType.BROKEN, LineType.SIDE])
        other_v_spawn_road = Road(self.NODE_2, self.NODE_3)
        # other_v_spawn_road2 = Road(self.NODE_2, self.NODE_3)
        # print("other_v_spawn_road2: "+str(other_v_spawn_road2))

        CreateRoadFrom(
            next_lane,
            lane_num,
            other_v_spawn_road,
            self.block_network,
            self._global_network,
            center_line_type=LineType.NONE,
            inner_lane_line_type=LineType.NONE,
            # side_lane_line_type=LineType.NONE,
            ignore_intersection_checking=self.ignore_intersection_checking
        )

        adverse_road = Road(self.NODE_3, self.NODE_2)
        lanes = get_lanes_on_road(other_v_spawn_road, self.block_network)
        print("lanes: " + str(lanes))
        print("adverse_road: " + str(adverse_road))
        print("other_v_spawn_road: " + str(other_v_spawn_road))
        reference_lane = lanes[-1]
        num = len(lanes)
        width = reference_lane.width_at(0)
        assert isinstance(reference_lane, StraightLane)
        start_point = reference_lane.position(lanes[-1].length, -(num - 1) * width)
        end_point = reference_lane.position(0, -(num - 1) * width)
        print("startpoint: " + str(start_point))
        print("endpoint: " + str(end_point))
        symmetric_lane = StraightLane(
            start_point, end_point, width, lanes[-1].line_types, reference_lane.forbidden,
            reference_lane.speed_limit,
            reference_lane.priority
        )

        print("symmetric_lane: " + str(symmetric_lane))
        CreateRoadFrom(
            symmetric_lane,
            num,
            adverse_road,
            self.block_network,
            self._global_network,
            # ignore_start=ignore_start,
            # ignore_end=ignore_end,
            # side_lane_line_type=side_lane_line_type,
            inner_lane_line_type=LineType.NONE,
            center_line_type=LineType.NONE,
            ignore_intersection_checking=self.ignore_intersection_checking
        )
        # -------------------------------------------------------------------
        next_lane1 = ExtendStraightLane(next_lane, self.BOT_LENGTH, [LineType.BROKEN, LineType.SIDE])
        start_point = next_lane1.start
        end_point = next_lane1.end
        width = next_lane1.width_at(0)  # - 9
        next_lane = StraightLane(
            start_point, end_point, width, next_lane1.line_types, next_lane1.forbidden, next_lane1.speed_limit,
            next_lane1.priority
        )
        print("reference_lane 1_width: " + str(next_lane.width_at(0)))

        other_v_spawn_road = Road(self.NODE_3, self.NODE_4)
        # other_v_spawn_road2 = Road(self.NODE_2, self.NODE_3)
        # print("other_v_spawn_road2: "+str(other_v_spawn_road2))

        CreateRoadFrom(
            next_lane,
            lane_num,
            other_v_spawn_road,
            self.block_network,
            self._global_network,
            center_line_type=LineType.NONE,
            inner_lane_line_type=LineType.NONE,
            # side_lane_line_type=LineType.NONE,
            ignore_intersection_checking=self.ignore_intersection_checking
        )

        adverse_road = Road(self.NODE_4, self.NODE_3)
        lanes = get_lanes_on_road(other_v_spawn_road, self.block_network)
        print("lanes: " + str(lanes))
        print("adverse_road: " + str(adverse_road))
        print("other_v_spawn_road: " + str(other_v_spawn_road))
        reference_lane = lanes[-1]
        num = len(lanes)
        width = reference_lane.width_at(0)
        if isinstance(reference_lane, StraightLane):
            start_point = reference_lane.position(lanes[-1].length, -(num - 1) * width)
            end_point = reference_lane.position(0, -(num - 1) * width)
            print("startpoint: " + str(start_point))
            print("endpoint: " + str(end_point))
            symmetric_lane = StraightLane(
                start_point, end_point, width, lanes[-1].line_types, reference_lane.forbidden,
                reference_lane.speed_limit,
                reference_lane.priority
            )

        print("symmetric_lane: " + str(symmetric_lane))
        CreateRoadFrom(
            symmetric_lane,
            num,
            adverse_road,
            self.block_network,
            self._global_network,
            inner_lane_line_type=LineType.NONE,
            center_line_type=LineType.NONE,
            ignore_intersection_checking=self.ignore_intersection_checking
        )

        print("bottleneck end")
        # ----------------------------------------------------------------
        next_lane1 = ExtendStraightLane(next_lane, self.END_LENGTH, [LineType.BROKEN, LineType.SIDE])
        start_point = next_lane1.start
        end_point = next_lane1.end
        width = next_lane1.width_at(0)  # + 9
        next_lane = StraightLane(
            start_point, end_point, width, next_lane1.line_types, next_lane1.forbidden, next_lane1.speed_limit,
            next_lane1.priority
        )
        print("reference_lane 2_width: " + str(next_lane.width_at(0)))

        other_v_spawn_road = Road(self.NODE_4, self.NODE_5)

        CreateRoadFrom(
            next_lane,
            lane_num,
            other_v_spawn_road,
            self.block_network,
            self._global_network,
            center_line_type=LineType.NONE,
            inner_lane_line_type=LineType.NONE,
            # side_lane_line_type=LineType.NONE,
            ignore_intersection_checking=self.ignore_intersection_checking
        )

        adverse_road = Road(self.NODE_5, self.NODE_4)
        lanes = get_lanes_on_road(other_v_spawn_road, self.block_network)
        print("lanes: " + str(lanes))
        print("adverse_road: " + str(adverse_road))
        print("other_v_spawn_road: " + str(other_v_spawn_road))
        reference_lane = lanes[-1]
        num = len(lanes)
        width = reference_lane.width_at(0)
        if isinstance(reference_lane, StraightLane):
            start_point = reference_lane.position(lanes[-1].length, -(num - 1) * width)
            end_point = reference_lane.position(0, -(num - 1) * width)
            print("startpoint: " + str(start_point))
            print("endpoint: " + str(end_point))
            symmetric_lane = StraightLane(
                start_point, end_point, width, lanes[-1].line_types, reference_lane.forbidden,
                reference_lane.speed_limit,
                reference_lane.priority
            )

        print("symmetric_lane: " + str(symmetric_lane))
        CreateRoadFrom(
            symmetric_lane,
            num,
            adverse_road,
            self.block_network,
            self._global_network,
            inner_lane_line_type=LineType.NONE,
            center_line_type=LineType.NONE,
            ignore_intersection_checking=self.ignore_intersection_checking
        )

        print("Road end")

        # ---------------------------------------------------------------
        self._create_in_world()

        # global_network += self.block_network

        # We allow intersection!
        global_network.add(self.block_network, no_intersect=False)

        # pos_roads = BaseBlock.get_roads(self.block_network, direction='positive')
        # neg_roads = BaseBlock.get_roads(self.block_network, direction='negative')
        # print("positive roads list: " + str(pos_roads))
        # print("Negative roads list: " + str(neg_roads))

        socket = self.create_socket_from_positive_road(other_v_spawn_road)
        socket.set_index(self.name, 0)
        # print(socket)

        self.add_sockets(socket)

        pl1 = socket.positive_road.get_lanes(self._global_network)
        print("get positive_lanes: " + str(pl1))
        pl = socket.negative_road
        print("negative lanes: " + str(pl))

        self.attach_to_world(render_root_np, physics_world)
        self._respawn_roads = [other_v_spawn_road]


MABottleneckConfig = dict(
    spawn_roads=[Road(SecondPGBlock.NODE_2, SecondPGBlock.NODE_3)],
    num_agents=2,
    map_config=dict(exit_length=60, bottle_lane_num=1, neck_lane_num=1, neck_length=20, lane_width=3, lane_num=1),
    cross_yellow_line_done=False,
    vehicle_config={
        # "show_lidar": True,
        # "show_side_detector": True,
        # "show_lane_line_detector": True,
        "side_detector": dict(num_lasers=4, distance=20),  # laser num, distance
        "lane_line_detector": dict(num_lasers=4, distance=20),
        "lidar": {'num_lasers': 72, 'distance': 30, 'num_others': 0, 'gaussian_noise': 0.0, 'dropout_prob': 0.0}
    }  # laser num, distance
)


class MABidirectioinalMap(PGMap):
    def _generate(self):
        length = self.config["exit_length"]

        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"

        # Build a Second-block
        second_block = SecondPGBlock(
            self.road_network,
            self.config[self.LANE_WIDTH],
            self.config["bottle_lane_num"],
            parent_node_path,
            physics_world,
            length=length
        )
        self.blocks.append(second_block)
        print("*** Second block Done ***")
        # Build Bottleneck


class MABidirectionalMapManager(MapManager):
    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(MABidirectioinalMap, map_config=config["map_config"], random_seed=None)
        else:
            assert len(self.spawned_objects) == 1, "It is supposed to contain one map in this manager"
            _map = self.spawned_objects.values()[0]
        self.load_map(_map)
        self.current_map.spawn_roads = config["spawn_roads"]


class MultiAgentBidirectional(MultiAgentBottleneckEnv):
    @staticmethod
    def default_config() -> Config:
        MABottleneckConfig["map_config"]["lane_num"] = MABottleneckConfig["map_config"]["bottle_lane_num"]
        return MultiAgentBottleneckEnv.default_config().update(MABottleneckConfig, allow_add_new_key=True)

    def setup_engine(self):
        super(MultiAgentBidirectional, self).setup_engine()
        self.engine.update_manager("map_manager", MABidirectionalMapManager())

    def _respawn_single_vehicle(self, randomize_position=False):
        """
        Arbitrary insert a new vehicle to a new spawn place if possible.
        """
        safe_places_dict = self.engine.spawn_manager.get_available_respawn_places(
            self.current_map, randomize=randomize_position
        )
        if len(safe_places_dict) == 0:
            return None, None, None

        safe_places_dict = {'>>|>>>|0|0': {'identifier': '>>|>>>|0|0', 'config': {'destination_node': ('1y0_1_'),
                                                                                  'spawn_lane_index': ('>>', '>>>', 0),
                                                                                  'spawn_longitude': 4.0,
                                                                                  'spawn_lateral': 0},
                                           'spawn_point_heading': 0.0, 'spawn_point_position': (14.0, 0.0)},
                            '1y0_1_|1y0_0_|0|0': {'identifier': '1y0_1_|1y0_0_|0|0',
                                                  'config': {'destination_node': ('>'),
                                                             'spawn_lane_index': ('1y0_1_', '1y0_0_', 0),
                                                             'spawn_longitude': 0.0, 'spawn_lateral': 0},
                                                  'spawn_point_heading': 0.0, 'spawn_point_position': (14.0, 0.0)}}

        # '1y0_0_|>>>|0|0': {'identifier': '1y0_0_|>>>|0|0', 'config': {'destination_node': ('>'),
        # 'spawn_lane_index': ('1y0_0_', '>>>', 0), 'spawn_longitude': 0.0, 'spawn_lateral': 0},
        # 'spawn_point_heading': 0.0, 'spawn_point_position': (14.0, 0.0)} }

        # print("Safe Spawn places: "+str(safe_places_dict))
        # file = open("car.txt", "a")
        # file.write("Safe_places_dict = " + str(safe_places_dict)+" list(safe_places_dict.keys(): " + str(list(
        # safe_places_dict.keys())))
        # file.write("list(safe_places_dict.keys(): " + str(list(safe_places_dict.keys()))
        # file.close()

        born_place_index = get_np_random(self._DEBUG_RANDOM_SEED).choice(list(safe_places_dict.keys()), 1)[0]
        new_spawn_place = safe_places_dict[born_place_index]
        # file = open("car1.txt", "w")
        # file.write("born_place_index = " + str(born_place_index)+" new_spawn_place: " + str(new_spawn_place))
        # file.close()

        # born_place_index = get_np_random(self._DEBUG_RANDOM_SEED).choice(list(safe_places_dict.keys()), 1)[0]
        # new_spawn_place = safe_places_dict[born_place_index]

        new_agent_id, vehicle, step_info = self.agent_manager.propose_new_vehicle()
        # new_spawn_place_config = new_spawn_place["config"]
        # new_spawn_place_config = self.engine.spawn_manager.update_destination_for(new_agent_id,
        # new_spawn_place_config)
        # vehicle.config.update(new_spawn_place_config)

        new_spawn_place_config = new_spawn_place["config"]
        '''
        cho = randint(0, 1)
        if cho == 0:
            new_spawn_place_config = {'spawn_lane_index': ('1y0_1_', '1y0_0_', 0), 'destination_node': ('>>'), 
            'spawn_longitude': 4.0, 'spawn_lateral': 0}
            cho = 1 
        else:
            new_spawn_place_config = {'spawn_lane_index': ('>', '>>', 0), 'destination_node': ('1y0_0_'), 
            'spawn_longitude': 0.0, 'spawn_lateral': 0}
            cho = 0
        '''
        print("new_spawn_place_config:" + str(new_spawn_place_config))
        new_spawn_place_config = self.engine.spawn_manager.update_destination_for(new_agent_id, new_spawn_place_config)
        vehicle.config.update(new_spawn_place_config)

        vehicle.reset()
        after_step_info = vehicle.after_step()
        step_info.update(after_step_info)
        self.dones[new_agent_id] = False  # Put it in the internal dead-tracking dict.

        new_obs = self.observations[new_agent_id].observe(vehicle)
        return new_agent_id, new_obs, step_info


def _vis(render=True):
    env = MultiAgentBidirectional(
        {
            "horizon": 100000,
            "vehicle_config": {
                "lidar": {
                    "num_lasers": 72,
                    "num_others": 0,
                    "distance": 40
                },
                "show_lidar": False,
            },
            "use_render": render,
            # "debug": True,
            "manual_control": render,
            "num_agents": 2,

            # "debug_physics_world": True,
            "debug": True,
        }
    )
    o = env.reset()
    total_r = 0
    ep_s = 0
    for i in range(1, 100000):
        o, r, d, info = env.step({k: [1.0, .0] for k in env.vehicles.keys()})
        for r_ in r.values():
            total_r += r_
        ep_s += 1
        # d.update({"total_r": total_r, "episode length": ep_s})
        if render:
            render_text = {
                "total_r": total_r,
                "episode length": ep_s,
                "cam_x": env.main_camera.camera_x,
                "cam_y": env.main_camera.camera_y,
                "cam_z": env.main_camera.top_down_camera_height,
                "current_track_v": env.agent_manager.object_to_agent(env.current_track_vehicle.name)
            }
            track_v = env.agent_manager.object_to_agent(env.current_track_vehicle.name)
            render_text["tack_v_reward"] = r[track_v]
            render_text["dist_to_right"] = env.current_track_vehicle.dist_to_right_side
            render_text["dist_to_left"] = env.current_track_vehicle.dist_to_left_side
            env.render(text=render_text)
        if d["__all__"]:
            print(
                "Finish! Current step {}. Group Reward: {}. Average reward: {}".format(
                    i, total_r, total_r / env.agent_manager.next_agent_count
                )
            )
            # break
        if len(env.vehicles) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()


if __name__ == "__main__":
    # _draw()
    _vis(render=True)
    # _vis_debug_respawn()
    # _profile()
    # _long_run()
    # pygame_replay("bottle")
