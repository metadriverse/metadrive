# launch sockets to send sensor readings to ROS
import argparse
import struct

import numpy as np
import zmq

from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy


class RosSocketServer():

    def __init__(self):
        self.context = zmq.Context().instance()
        self.context.setsockopt(zmq.IO_THREADS, 2)
        self.img_socket = self.context.socket(zmq.PUSH)
        self.img_socket.setsockopt(zmq.SNDBUF, 4194304)
        self.img_socket.bind("ipc:///tmp/rgb_camera")
        self.img_socket.set_hwm(5)
        self.lidar_socket = self.context.socket(zmq.PUSH)
        self.lidar_socket.setsockopt(zmq.SNDBUF, 4194304)
        self.lidar_socket.bind("ipc:///tmp/lidar")
        self.lidar_socket.set_hwm(5)
        self.obj_socket = self.context.socket(zmq.PUSH)
        self.obj_socket.setsockopt(zmq.SNDBUF, 4194304)
        self.obj_socket.bind("ipc:///tmp/obj")
        self.obj_socket.set_hwm(5)

    def run(self, test=False):
        config = dict(
            use_render=True if not test else False,  # need this on to get the camera
            num_scenarios=1,
            horizon=1000,
            image_observation=True if not test else False,
            manual_control=False,
            agent_policy=ReplayEgoCarPolicy,
            norm_pixel=False,
            show_logo=False,
            show_fps=False,
            show_interface=False,
            physics_world_step_size=0.02,
            vehicle_config=dict(
                image_source="main_camera",
                show_navi_mark=False,
            ),
            data_directory=AssetLoader.file_path("nuscenes", unix_style=False),
        )

        env = ScenarioEnv(config)

        try:

            env.reset()
            print(HELP_MESSAGE)
            env.agent.expert_takeover = False
            while True:
                o = env.step([0, 0])
                if test:
                    image_data = np.zeros((512, 512, 3))  # fake data for testing
                    image_data[::16, :, :] = 255
                else:
                    image_data = o[0]['image'][..., -1]
                # send via socket
                image_data = image_data.astype(np.uint8)
                dim_data = struct.pack('ii', image_data.shape[1], image_data.shape[0])
                image_data = bytearray(image_data)
                # concatenate the dimensions and image data into a single byte array
                image_data = dim_data + image_data
                try:
                    self.img_socket.send(image_data, zmq.NOBLOCK)
                except zmq.error.Again:
                    msg = "ros_socket_server: error sending image"
                    if test:
                        raise ValueError(msg)
                    else:
                        print(msg)
                del image_data  # explicit delete to free memory

                lidar_data, objs = env.agent.lidar.perceive(
                    env.agent,
                    env.engine.physics_world.dynamic_world,
                    env.agent.config["lidar"]["num_lasers"],
                    env.agent.config["lidar"]["distance"],
                    height=1.0,
                )

                ego_x = env.agent.position[0]
                ego_y = env.agent.position[1]
                ego_theta = np.arctan2(env.agent.heading[1], env.agent.heading[0])

                num_data = struct.pack('i', len(objs))
                obj_data = []
                for obj in objs:
                    obj_x = obj.position[0]
                    obj_y = obj.position[1]
                    obj_theta = np.arctan2(obj.heading[1], obj.heading[0])

                    obj_x = obj_x - ego_x
                    obj_y = obj_y - ego_y
                    obj_x_new = np.cos(-ego_theta) * obj_x - np.sin(-ego_theta) * obj_y
                    obj_y_new = np.sin(-ego_theta) * obj_x + np.cos(-ego_theta) * obj_y

                    obj_data.append(obj_x_new)
                    obj_data.append(obj_y_new)
                    obj_data.append(obj_theta - ego_theta)
                    obj_data.append(obj.LENGTH)
                    obj_data.append(obj.WIDTH)
                    obj_data.append(obj.HEIGHT)
                obj_data = np.array(obj_data, dtype=np.float32)
                obj_data = bytearray(obj_data)
                obj_data = num_data + obj_data
                try:
                    self.obj_socket.send(obj_data, zmq.NOBLOCK)
                except zmq.error.Again:
                    msg = "ros_socket_server: error sending objs"
                    if test:
                        raise ValueError(msg)
                    else:
                        print(msg)
                del obj_data  # explicit delete to free memory

                # convert lidar data to xyz
                lidar_data = np.array(lidar_data) * env.agent.config["lidar"]["distance"]
                lidar_range = env.agent.lidar._get_lidar_range(
                    env.agent.config["lidar"]["num_lasers"], env.agent.lidar.start_phase_offset
                )
                point_x = lidar_data * np.cos(lidar_range)
                point_y = lidar_data * np.sin(lidar_range)
                point_z = np.ones(lidar_data.shape)  # assume height = 1.0
                lidar_data = np.stack([point_x, point_y, point_z], axis=-1).astype(np.float32)
                dim_data = struct.pack('i', len(lidar_data))
                lidar_data = bytearray(lidar_data)
                # concatenate the dimensions and lidar data into a single byte array
                lidar_data = dim_data + lidar_data
                try:
                    self.lidar_socket.send(lidar_data, zmq.NOBLOCK)
                except zmq.error.Again:
                    msg = "ros_socket_server: error sending lidar"
                    if test:
                        raise ValueError(msg)
                    else:
                        print(msg)
                del lidar_data  # explicit delete to free memory

                if o[2]:  # done
                    break

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

        finally:
            env.close()


def main(test=False):
    server = RosSocketServer()
    server.run(test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    main(args.test)
