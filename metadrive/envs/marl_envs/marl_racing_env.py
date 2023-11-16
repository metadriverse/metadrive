from collections import defaultdict

from metadrive.component.map.pg_map import PGMap
from metadrive.component.pg_space import Parameter
from metadrive.component.pgblock.curve import CurveWithGuardrail
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.straight import StraightWithGuardrail
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.utils import Config

RACING_CONFIG = dict(

    # Misc.
    use_render=False,
    manual_control=False,
    traffic_density=0,
    random_agent_model=False,
    top_down_camera_initial_x=95,
    top_down_camera_initial_y=15,
    top_down_camera_initial_z=120,
    allow_respawn=False,
    vehicle_config=dict(show_lidar=False, show_navi_mark=False),

    # Number of agents and map setting.
    num_agents=8,
    map_config=dict(lane_num=8, exit_length=20),

    # Reward setting
    use_lateral=False,

    # Termination condition
    cross_yellow_line_done=True,
    out_of_road_done=True,
    on_continuous_line_done=False,
    out_of_route_done=False,
    crash_done=False,
    max_step_per_agent=5_000,
    horizon=5_000,

    # Debug setting
    # agent_policy=IDMPolicy,
)


class RacingMap(PGMap):
    def _generate(self):
        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"

        # FIRST_BLOCK_LENGTH = 100

        init_block = FirstPGBlock(
            self.road_network,
            lane_width=self.config[PGMap.LANE_WIDTH],
            lane_num=self.config[PGMap.LANE_NUM],
            render_root_np=parent_node_path,
            physics_world=physics_world,
            ignore_adverse_road=True,
        )
        self.blocks.append(init_block)

        curve_direction = 1
        curve_len = 100

        block_c1 = CurveWithGuardrail(1, init_block.get_socket(0), self.road_network, random_seed=1)
        block_c1.construct_from_config(
            {
                Parameter.length: curve_len,
                Parameter.radius: 50,
                Parameter.angle: 90,
                Parameter.dir: curve_direction,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c1)

        block_s2 = StraightWithGuardrail(
            2,
            block_c1.get_socket(0),
            self.road_network,
            random_seed=1,
        )
        block_s2.construct_from_config({
            Parameter.length: 5,
        }, parent_node_path, physics_world)
        self.blocks.append(block_s2)

        block_c2 = CurveWithGuardrail(3, block_s2.get_socket(0), self.road_network, random_seed=1)
        block_c2.construct_from_config(
            {
                Parameter.length: curve_len,
                Parameter.radius: 50,
                Parameter.angle: 90,
                Parameter.dir: curve_direction,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c2)

        block_c3 = CurveWithGuardrail(4, block_c2.get_socket(0), self.road_network, random_seed=1)
        block_c3.construct_from_config(
            {
                Parameter.length: curve_len,
                Parameter.radius: 50,
                Parameter.angle: 90,
                Parameter.dir: curve_direction,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_c3)

        block_s3 = StraightWithGuardrail(5, block_c3.get_socket(0), self.road_network, random_seed=1)
        block_s3.construct_from_config({
            Parameter.length: 5,
        }, parent_node_path, physics_world)
        self.blocks.append(block_s3)

        block_c4 = CurveWithGuardrail(6, block_s3.get_socket(0), self.road_network, random_seed=1)
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._completed_loop_count = defaultdict(int)

    def setup_engine(self):
        super(MultiAgentRacingEnv, self).setup_engine()
        self.engine.update_manager("map_manager", RacingMapManager())

    @staticmethod
    def default_config() -> Config:
        return MultiAgentMetaDrive.default_config().update(RACING_CONFIG, allow_add_new_key=True)

    @property
    def real_destination(self):
        return list(self.current_map.road_network.graph.keys())[-1]

    @property
    def fake_destination(self):
        return list(self.current_map.road_network.graph.keys())[-2]

    def _is_arrive_destination(self, vehicle):
        """
        Args:
            vehicle: The BaseVehicle instance.

        Returns:
            flag: Whether this vehicle arrives its destination.
        """
        flag = super()._is_arrive_destination(vehicle)
        if flag:
            if vehicle.config["destination"] == self.fake_destination:
                vehicle.config["destination"] = self.real_destination
            else:
                vehicle.config["destination"] = self.fake_destination
            vehicle.reset_navigation()
            flag = False
        return flag

    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        reward = 0.0
        reward += self.config["driving_reward"] * (long_now - long_last)
        reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h)

        step_info["step_reward"] = reward

        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        return reward, step_info


def _vis():
    env = MultiAgentRacingEnv()
    o, _ = env.reset()
    total_r = 0
    ep_s = 0
    for i in range(1, 100000):
        o, r, tm, tc, info = env.step({k: [-0.05, 1.0] for k in env.vehicles.keys()})
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
            total_r = 0
            print("Reset")
            env.reset()
    env.close()


if __name__ == "__main__":
    _vis()
