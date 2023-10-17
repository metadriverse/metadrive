# a simple script to send images via socket
import random
import zmq
import struct
import numpy as np

from metadrive import MetaDriveEnv
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.constants import HELP_MESSAGE

class myBridge():
  def __init__(self):
    #self.params.remove("CalibrationParams")
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
      

  def game_runner(self):
    config = dict(
      use_render=True, # need this on to get the camera
      num_scenarios=1,
      horizon = 1000,
      image_observation=True,
      image_on_cuda=True,
      manual_control=True,
      rgb_clip=False,
      # random_lane_width=True,
      # random_lane_num=True,
      # random_agent_model=False,
      # traffic_density=0.5,
      # camera_dist= 0.0,
      # camera_pitch= 0,
      # camera_height= 0.6,
      # camera_smooth= False,
      show_interface= False,
      physics_world_step_size = 0.02, # this means the actual fps for the camera is 0.1s
      vehicle_config = dict(image_source="rgb_camera", 
                            rgb_camera="bridges/sensor_configs/test_camera_parameter.json",
                            stack_size=1,
                            show_navi_mark=False,
                            ),
      data_directory = AssetLoader.file_path("nuscenes", return_raw_style=False),
    )

    env = ScenarioEnv(config)
    #rk = Ratekeeper(4, print_delay_threshold=None)
    try:

      env.reset()
      print(HELP_MESSAGE)
      env.vehicle.expert_takeover = False
      while True:
        o,_,_,_ = env.step([0,0])
        image_data = o["image"].get()[..., -1]
        #send via socket
        image_data = image_data.astype(np.uint8)
        # import cv2
        # cv2.imshow("window", data)
        # cv2.waitKey(1)
        dim_data = struct.pack('ii', image_data.shape[1], image_data.shape[0])
        image_data = bytearray(image_data)
        # concatenate the dimensions and image data into a single byte array
        image_data = dim_data + image_data
        try:
            self.img_socket.send(image_data, zmq.NOBLOCK)
        except zmq.error.Again:
            print("ros_socket_server: error sending image")
        del image_data # explicit delete to free memory

        lidar_data, objs = env.vehicle.lidar.perceive(env.vehicle)

        ego_x = env.vehicle.position[0]
        ego_y = env.vehicle.position[1]
        ego_theta = np.arctan2(env.vehicle.heading[1], env.vehicle.heading[0])

        num_data = struct.pack('i', len(objs))
        obj_data = []
        for obj in objs:
          obj_x = obj.position[0]
          obj_y = obj.position[1]
          obj_theta = np.arctan2(obj.heading[1], obj.heading[0])
          obj_data.append(obj_x-ego_x)
          obj_data.append(obj_y-ego_y)
          obj_data.append(obj_theta-ego_theta)
          obj_data.append(obj.LENGTH)
          obj_data.append(obj.WIDTH)
          obj_data.append(obj.HEIGHT)
        obj_data = np.array(obj_data, dtype=np.float32)
        obj_data = bytearray(obj_data)
        obj_data = num_data + obj_data
        try:
            self.obj_socket.send(obj_data, zmq.NOBLOCK)
        except zmq.error.Again:
            print("ros_socket_server: error sending objs")
        del obj_data # explicit delete to free memory
      
        # convert lidar data to xyz
        lidar_data = np.array(lidar_data) * env.vehicle.lidar.perceive_distance
        point_x = lidar_data * np.cos(env.vehicle.lidar._lidar_range)
        point_y = lidar_data * np.sin(env.vehicle.lidar._lidar_range)
        point_z = np.ones(lidar_data.shape)*env.vehicle.lidar.height
        lidar_data = np.stack([point_x, point_y, point_z], axis=-1).astype(np.float32)
        dim_data = struct.pack('i', len(lidar_data))
        lidar_data = bytearray(lidar_data)
        # concatenate the dimensions and image data into a single byte array
        lidar_data = dim_data + lidar_data
        try:
            self.lidar_socket.send(lidar_data, zmq.NOBLOCK)
        except zmq.error.Again:
            print("ros_socket_server: error sending lidar")
        del lidar_data # explicit delete to free memory


        
            
    except Exception as e:
      import traceback
      traceback.print_exc()
      raise e

    finally:
      env.close()
        
def main():
  bridge = myBridge() 
  bridge.game_runner()  

if __name__ == "__main__":
  main()