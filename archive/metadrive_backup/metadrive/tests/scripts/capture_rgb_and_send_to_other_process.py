"""
Script from https://github.com/metadriverse/metadrive/issues/290
Need to install zmq and lz4
"""
import zmq
import lz4.frame as lz4
import argparse
import random
from metadrive import MetaDriveEnv
from metadrive.constants import HELP_MESSAGE
import threading
import numpy as np
import cv2

W, H = 1920, 1200


def main_thread():
    config = dict(
        use_render=True,
        image_observation=True,
        manual_control=True,  # set false for external subscriber control
        traffic_density=0.0,
        num_scenarios=100,
        random_agent_model=True,
        random_lane_width=True,
        random_lane_num=True,
        vehicle_config=dict(image_source="rgb_camera", rgb_camera=(W, H), stack_size=1),
        map=4,  # seven block
        start_seed=random.randint(0, 1000),
        window_size=(300, 200)
    )
    env = MetaDriveEnv(config)
    try:
        o, _ = env.reset()
        # print(HELP_MESSAGE)
        env.agent.expert_takeover = False
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.bind("tcp://127.0.0.1:5555")
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.setsockopt(zmq.RCVHWM, 1)
        vehicle_socket = context.socket(zmq.PUSH)
        vehicle_socket.bind("tcp://127.0.0.1:5556")  # vehicle state
        vehicle_socket.setsockopt(zmq.SNDHWM, 1)
        vehicle_socket.setsockopt(zmq.RCVHWM, 1)

        assert isinstance(o, dict)
        throttle_brake_op = 2.0
        steer_op = 0.0
        while True:
            # sm.update(0) # cereal submaster
            # throttle_brake_op = sm['carControl'].actuators.accel
            # steer_op = sm['carControl'].actuators.steeringAngleDeg
            o, rewards, dones, step_infos = env.step([steer_op, throttle_brake_op])
            try:
                img = o['image']  # (W,H,3,stack_size)
                compressed_image = lz4.compress(img)
                socket.send(compressed_image, zmq.NOBLOCK)
            except zmq.error.Again:
                # print("Dropped frame")
                pass
            try:
                vehicle_socket.send_pyobj(step_infos, zmq.NOBLOCK)
            except zmq.error.Again:
                # print("Dropped vehicle state")
                pass
    except Exception as e:
        raise e
    finally:
        env.close()
        socket.close()
        context.term()


if __name__ == "__main__":
    main_thread()
