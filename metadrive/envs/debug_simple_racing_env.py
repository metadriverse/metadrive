import argparse

from metadrive import MetaDriveEnv
from metadrive.component.map.pg_map import PGMap
from metadrive.component.pg_space import Parameter
from metadrive.component.pgblock.curve import CurveWithGuardrail
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.straight import StraightWithGuardrail
from metadrive.constants import HELP_MESSAGE
from metadrive.manager.pg_map_manager import PGMapManager


class RacingEnv(MetaDriveEnv):
    def setup_engine(self):
        super(RacingEnv, self).setup_engine()
        self.engine.update_manager("map_manager", RacingMapManager())

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

            if (
                    len(complete_checkpoints) - 1 > complete_checkpoints.index(current_lane_index) >=
                len(complete_checkpoints) - 3
            ) and not changed_dest:
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
