"""
This script demonstrates how to setup the Multi-agent RL environments.


Usage: python -m metadrive.examples.drive_in_multi_agent_env --env pgma

Options for --env argument:
    (1) roundabout
    (2) intersection
    (3) tollgate
    (4) bottleneck
    (5) parkinglot
    (6) pgma

"""
import numpy as np
import argparse
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive import (
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv, MultiAgentParkingLotEnv
)
from metadrive.constants import HELP_MESSAGE
from metadrive.policy.idm_policy import ManualControllableIDMPolicy, ManualControllableFreezePolicy
from metadrive.component.sensors.lidar import LidarGroup


DEFAULT_LIDAR_CONFIG = [
            dict(
                num_lasers = 10,
                distance = 10,
                enable_show = True,
                hfov = 60,
                vfov = 5,
                pos_offset= (1,1),
                angle_offset= 15,
                pitch = 0,
                num_lasers_v = 5
            ),
            dict(
                num_lasers = 10,
                distance = 10,
                enable_show = True,
                hfov = 60,
                vfov = 5,
                pos_offset= (1,-1),
                angle_offset= 285,
                pitch = 0,
                num_lasers_v = 5
            ),
            dict(
                num_lasers = 10,
                distance = 10,
                enable_show = True,
                hfov = 60,
                vfov = 5,
                pos_offset= (-1,-1),
                angle_offset= 195,
                pitch = 0,
                num_lasers_v = 5
            ),
            dict(
                num_lasers = 10,
                distance = 10,
                enable_show = True,
                hfov = 60,
                vfov = 5,
                pos_offset= (-1,1),
                angle_offset= 105,
                pitch = 0,
                num_lasers_v = 5
            )
        ]
import json

if __name__ == "__main__":
    envs = dict(
        roundabout=MultiAgentRoundaboutEnv,
        intersection=MultiAgentIntersectionEnv,
        tollgate=MultiAgentTollgateEnv,
        bottleneck=MultiAgentBottleneckEnv,
        parkinglot=MultiAgentParkingLotEnv,
        pgma=MultiAgentMetaDrive
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="parkinglot", choices=list(envs.keys()))
    parser.add_argument("--top_down", action="store_true")
    args = parser.parse_args()
    env_cls_name = args.env
    extra_args = dict(film_size=(800, 800)) if args.top_down else {}
    assert env_cls_name in envs.keys(), "No environment named {}, argument accepted: \n" \
                                        "(1) roundabout\n" \
                                        "(2) intersection\n" \
                                        "(3) tollgate\n" \
                                        "(4) bottleneck\n" \
                                        "(5) parkinglot\n" \
                                        "(6) pgma" \
        .format(env_cls_name)
    env = envs[env_cls_name](
        {
            "use_render": True if not args.top_down else False,
            "crash_done": False,
            "sensors": dict(rgb_camera=(RGBCamera, 512, 256)),
            "interface_panel": ["rgb_camera","dashboard"],
            "agent_policy": ManualControllableFreezePolicy,
            "vehicle_config": dict(show_lidar=False, show_navi_mark=False),
        }
    )
    try:
        env.reset()
        # if env.current_track_vehicle:
        #     env.current_track_vehicle.expert_takeover = True
        print(HELP_MESSAGE)
        env.switch_to_third_person_view()  # Default is in Top-down view, we switch to Third-person view.
        buffer = {}
        for i in range(1, 10):
            scene_identifier =  env.current_seed
            o, r, tm, tc, info = env.step({agent_id: [0, 0] for agent_id in env.vehicles.keys()})
            lidar_group = env.current_track_vehicle.lidar
            
            num_lidar = len(lidar_group.lidars)
            distance = np.array([lidar_group.lidars[i].perceive_distance for i in range(num_lidar)])
            closest_points = lidar_group.data
            mkey = None
            for id, vehicle in env.vehicles.items():
                if vehicle.id == env.current_track_vehicle.id:
                    mkey = id
            def get_points(closest_points):
                points = []
                for _, coord in closest_points:
                    if coord is None:
                        points.append((0,0,100))
                    else:
                        points.append((coord[0],coord[1],coord[2]))
                return points

            
            data_point = {
                "ego_pos": env.current_track_vehicle.position,
                "ego_heading": env.current_track_vehicle.heading,
                "ego_lidar_reading": (o[mkey][-num_lidar:]*distance).tolist(),
                "ground_truth_v3": get_points(closest_points), #Note, 0,0, 100 means there's no point detected within the range of that lidar
                "ego_state": (o[mkey][:-num_lidar]).tolist(),
                "scene_identifier":env.current_track_vehicle.id
            }
            buffer[i] = data_point
            env.render(
                **extra_args,
                mode="top_down" if args.top_down else None,
                text={
                    "Quit": "ESC",
                    "Number of existing vehicles": len(env.vehicles),
                    "Tracked agent (Press Q)": env.engine.agent_manager.object_to_agent(env.current_track_vehicle.id),
                    "Keyboard Control": "W,A,S,D",
                    # "Auto-Drive (Switch mode: T)": "on" if env.current_track_vehicle.expert_takeover else "off",
                } if not args.top_down else {}
            )
            if tm["__all__"]:
                env.reset()
                # if env.current_track_vehicle:
                #     env.current_track_vehicle.expert_takeover = True
        with open("data.json", "w") as file:
            json.dump(buffer, file)
            
    except Exception as e:
        raise e
    finally:
        env.close()
