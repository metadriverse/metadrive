import copy
from collections import defaultdict

from metadrive.component.map.pg_map import PGMap
from metadrive.component.pg_space import Parameter
from metadrive.component.pgblock.curve import CurveWithGuardrail
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.straight import StraightWithGuardrail
from metadrive.constants import TerminationState
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.utils import Config

RACING_CONFIG = dict(
    # controller="joystick",
    num_agents=2,
    use_render=False,
    manual_control=False,
    traffic_density=0,
    num_scenarios=10000,
    random_agent_model=False,
    top_down_camera_initial_x=95,
    top_down_camera_initial_y=15,
    top_down_camera_initial_z=120,
    # random_lane_width=True,
    # random_lane_num=True,
    cross_yellow_line_done=True,
    use_lateral=False,
    on_continuous_line_done=False,

    out_of_route_done=False,
    crash_done=False,

    vehicle_config=dict(show_lidar=False, show_navi_mark=False),
    agent_policy=IDMPolicy,
)


class RacingMap(PGMap):
    def _generate(self):
        FirstPGBlock.ENTRANCE_LENGTH = 0.5
        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"

        init_block = FirstPGBlock(self.road_network, 10.0, 3, parent_node_path, physics_world, 1)
        self.blocks.append(init_block)

        curve_direction = 1
        curve_len = 100

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
    def __init__(self):
        super(RacingMapManager, self).__init__()

    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(RacingMap, map_config=config["map_config"], random_seed=None)
        else:
            assert len(self.spawned_objects) == 1, "It is supposed to contain one map in this manager"
            _map = self.spawned_objects.values()[0]
        self.load_map(_map)


class MultiAgentRacingEnv(MultiAgentMetaDrive):
    NUM_LOOPS = 3  # TODO: Remove to config

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._completed_loop_count = defaultdict(int)

    def setup_engine(self):
        super(MultiAgentRacingEnv, self).setup_engine()
        self.engine.update_manager("map_manager", RacingMapManager())

    @staticmethod
    def default_config() -> Config:
        return MultiAgentMetaDrive.default_config().update(RACING_CONFIG, allow_add_new_key=True)

    def step(self, actions):
        o, r, terminated, truncated, i = super().step(actions=actions)

        # Hack in here to check and reset the navigation system for each agent.

        # for vid, vehicle in self.vehicles.items():
        #     current_lane_index = vehicle.lane_index[1]
        #     current_checkpoint_index = vehicle.navigation.checkpoints.index(current_lane_index)
        #     total_checkpoint = len(vehicle.navigation.checkpoints)
        #
        #     # If the vehicle is reaching the fake destination, we reset the destination to the real destination.
        #     if total_checkpoint - 1 > current_checkpoint_index and current_checkpoint_index >= total_checkpoint - 3:
        #         # Say you have 10 checkpoints, the real destination is the 10th checkpoint and the fake destination
        #         # is the 9th checkpoint. If current checkpoint is the 7th or 8th checkpoint, then
        #         # you are reaching the fake checkpoint.
        #         # In this case, we need to reset the destination to the real destination.
        #         vehicle.config["destination"] = self.real_destination
        #         vehicle.reset_navigation()
        #
        #     # If the vehicle reached the real destination, but it hasn't finished enough loops, we reset
        #     # the destination to the fake destination again.
        #     # Note that this part is already processed in the done function.
        #
        #     pass
        return o, r, terminated, truncated, i

    def done_function(self, vehicle_id):
        """
        In this environment, the termination can only happen in two cases: out of road, or success.

        Args:
            vehicle_id: (str) representing the vehicle's ID.

        Returns:
            done: (bool) whether we should terminate this agent.
        """

        # The single agent environment will investigate whether this agent is successful, out of road or crashing into
        # others.
        done, done_info = super(MultiAgentMetaDrive, self).done_function(vehicle_id)

        # We hack in the done_function to reset the navigation system of each vehicle if necessary.
        # This is helpful because this particular environment contains a cyclic map.
        vehicle = self.vehicles[vehicle_id]
        current_lane_index = vehicle.lane_index[1]

        checkpoints = vehicle.navigation.checkpoints

        current_checkpoint_index = checkpoints.index(current_lane_index)
        total_checkpoint = len(checkpoints)

        # If the vehicle is reaching the fake destination, we reset the destination to the real destination.
        if current_checkpoint_index + 1 >= total_checkpoint - 3:
            # Say you have 10 checkpoints, the real destination is the 10th checkpoint and the fake destination
            # is the 9th checkpoint. If current checkpoint is the 7th or 8th checkpoint, then
            # you are reaching the fake checkpoint.
            # In this case, we need to reset the destination to the real destination.
            vehicle.config["destination"] = self.real_destination
            vehicle.navigation.reset(vehicle)
            print(111)

        # If the vehicle reached the real destination, but it hasn't finished enough loops, we reset
        # the destination to the fake destination again.
        if done_info[TerminationState.SUCCESS]:
            if self._completed_loop_count[vehicle_id] < self.NUM_LOOPS:
                self._completed_loop_count[vehicle_id] += 1
                done_info[TerminationState.SUCCESS] = False
                done = False
                vehicle.config["destination"] = self.fake_destination
                vehicle.navigation.reset(vehicle)
            else:
                assert done

        return done, done_info

    def reset(self, seed=None):
        ret = super().reset(seed=seed)
        self.real_destination = list(self.current_map.road_network.graph.keys())[-1]
        self.fake_destination = list(self.current_map.road_network.graph.keys())[-2]

        # vehicle = self.vehicles["agent0"]
        # self.default_checkpoints = copy.deepcopy(vehicle.navigation.checkpoints)

        self.loop_counter = defaultdict(int)
        for vid, v in self.vehicles.items():
            v.config["destination"] = self.fake_destination
            v.reset_navigation()
        return ret

    def _is_arrive_destination(self, vehicle):
        ret = super()._is_arrive_destination(vehicle)
        if ret:
            vid = vehicle.id

            # Reset the navigation
            vehicle.config["destination"] = list(self.current_map.road_network.graph.keys())[-2]
            vehicle.navigation.reset(vehicle)

            self._completed_loop_count[vid] += 1
            if self._completed_loop_count[vid] >= self.NUM_LOOPS:
                return True
        return False

    # def reward_function(self, vehicle_id: str):
    #     """
    #     Override this func to get a new reward function
    #     :param vehicle_id: id of BaseVehicle
    #     :return: reward
    #     """
    #     vehicle = self.vehicles[vehicle_id]
    #     step_info = dict()
    #
    #     # Reward for moving forward in current lane
    #     if vehicle.lane in vehicle.navigation.current_ref_lanes:
    #         current_lane = vehicle.lane
    #     else:
    #         current_lane = vehicle.navigation.current_ref_lanes[0]
    #         current_road = vehicle.navigation.current_road
    #     long_last, _ = current_lane.local_coordinates(vehicle.last_position)
    #     long_now, lateral_now = current_lane.local_coordinates(vehicle.position)
    #
    #     # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
    #     if self.config["use_lateral"]:
    #         lateral_factor = clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
    #     else:
    #         lateral_factor = 1.0
    #     lateral_factor = 1.0
    #
    #     reward = 0.0
    #     reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor
    #     reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h)
    #
    #     step_info["step_reward"] = reward
    #
    #     if self._is_arrive_destination(vehicle):
    #         reward = +self.config["success_reward"]
    #     elif self._is_out_of_road(vehicle):
    #         reward = -self.config["out_of_road_penalty"]
    #     elif vehicle.crash_vehicle:
    #         reward = -self.config["crash_vehicle_penalty"]
    #     elif vehicle.crash_object:
    #         reward = -self.config["crash_object_penalty"]
    #     return reward, step_info
    #
    # def _is_out_of_road(self, vehicle):
    #     # A specified function to determine whether this vehicle should be done.
    #     # return vehicle.on_yellow_continuous_line or (not vehicle.on_lane) or vehicle.crash_sidewalk
    #     ret = vehicle.on_white_continuous_line or (not vehicle.on_lane) or vehicle.crash_sidewalk
    #     if self.config["cross_yellow_line_done"]:
    #         ret = ret or vehicle.on_yellow_continuous_line
    #     return ret


def _vis():
    env = MultiAgentRacingEnv(
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
            "use_render": False,
            # "debug": True,
            "manual_control": False,
            "num_agents": 2,
            "agent_policy": IDMPolicy
        }
    )
    o, _ = env.reset()
    total_r = 0
    ep_s = 0
    for i in range(1, 100000):
        o, r, tm, tc, info = env.step({k: [1.0, .0] for k in env.vehicles.keys()})

        for r_ in r.values():
            total_r += r_
        ep_s += 1
        env.render(mode="topdown")
        if tm["__all__"]:
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
    _vis()
