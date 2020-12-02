import gym
import numpy as np

from pg_drive.envs.observation_type import LidarStateObservation, ImageStateObservation
from pg_drive.pg_config.pg_config import PgConfig
from pg_drive.scene_creator.algorithm.BIG import BigGenerateMethod
from pg_drive.scene_creator.ego_vehicle.base_vehicle import BaseVehicle
from pg_drive.scene_creator.map import Map
from pg_drive.scene_manager.traffic_manager import TrafficManager, TrafficMode
from pg_drive.world.pg_world import PgWorld
from pg_drive.world.chase_camera import ChaseCamera
from pg_drive.world.manual_controller import KeyboardController, JoystickController


class GeneralizationRacing(gym.Env):
    def __init__(self, config: dict = None):
        self.config = self.default_config()
        if config:
            self.config.update(config)

        # set their value after vehicle created
        vehicle_config = BaseVehicle.get_vehicle_config(self.config["vehicle_config"])
        self.observation = LidarStateObservation(vehicle_config) if not self.config["use_rgb"] \
            else ImageStateObservation(vehicle_config, self.config["image_buffer_name"], self.config["rgb_clip"])
        self.observation_space = self.observation.observation_space
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2, ), dtype=np.float32)

        self.start_seed = self.config["start_seed"]
        self.env_num = self.config["environment_num"]
        self.map_config = self.config["map_config"]
        self.use_render = self.config["use_render"]
        pg_world_config = self.config["pg_world_config"]
        pg_world_config.update(
            {
                "use_render": self.use_render,
                "use_rgb": self.config["use_rgb"],
                "debug": self.config["debug"],
            }
        )
        self.pg_world_config = pg_world_config

        # lazy initialization, create the main vehicle in the lazy_init() func
        self.pg_world = None
        self.traffic_manager = None
        self.control_camera = None
        self.controller = None

        self.maps = {_seed: None for _seed in range(self.start_seed, self.start_seed + self.env_num)}
        self.current_seed = self.start_seed
        self.current_map = None
        self.vehicle = None
        self.done = False

    def lazy_init(self):
        # It is the true init() func to create the main vehicle and its module
        if self.pg_world is not None:
            return

        # init world
        self.pg_world = PgWorld(self.pg_world_config)

        # init traffic manager
        self.traffic_manager = TrafficManager(self.config["traffic_mode"])

        # for manual_control and camera type
        if self.config["use_chase_camera"]:
            self.control_camera = ChaseCamera(self.config["camera_height"], 7, self.pg_world)
        if self.config["manual_control"]:
            if self.config["controller"] == "keyboard":
                self.controller = KeyboardController()
            elif self.config["controller"] == "joystick":
                self.controller = JoystickController(self.pg_world)
            else:
                raise ValueError("No such a controller type: {}".format(self.config["controller"]))

        # init vehicle
        v_config = self.config["vehicle_config"]
        self.vehicle = BaseVehicle(self.pg_world, v_config)

        if self.use_render or self.config["use_rgb"]:
            self.control_camera.reset(self.vehicle.position)

    @staticmethod
    def default_config() -> PgConfig:
        env_config = dict(

            # ===== Rendering =====
            use_render=False,  # pop a window to render or not
            force_fps=None,
            debug=False,
            manual_control=False,
            controller="keyboard",  # "joystick" or "keyboard"
            use_chase_camera=True,
            camera_height=1.8,

            # ===== Traffic =====
            traffic_density=0.1,
            traffic_mode=TrafficMode.Add_once,

            # ===== Observation =====
            use_rgb=False,
            rgb_clip=True,
            vehicle_config=dict(),  # use default vehicle modules see more in BaseVehicle
            image_buffer_name="front_cam",  # mini_map or front_cam, the name must be as same as the module name

            # ===== Map Config =====
            map_config={
                Map.GENERATE_METHOD: BigGenerateMethod.BLOCK_NUM,
                Map.GENERATE_PARA: 3
            },

            # ===== Generalization =====
            start_seed=0,
            environment_num=1,

            # ===== Action =====
            decision_repeat=5,

            # ===== Reward Scheme =====
            success_reward=20,
            out_of_road_penalty=5,
            crash_penalty=10,
            acceleration_penalty=0.0,
            steering_penalty=0.1,
            low_speed_penalty=0.0,
            driving_reward=1.0,
            general_penalty=0.0,
            speed_reward=0.1,

            # ===== Others =====
            pg_world_config=dict(),
            use_increment_steering=False,
            action_check=False,
        )
        return PgConfig(env_config)

    def render(self, mode='human', text: dict = None):
        assert self.use_render or self.config["use_rgb"], "render is off now, can not render"
        if self.control_camera is not None:
            self.control_camera.renew_camera_place(self.pg_world.cam, self.vehicle)
        self.pg_world.render_frame(text)
        if self.pg_world.vehicle_panel is not None:
            self.pg_world.vehicle_panel.renew_2d_car_para_visualization(
                self.vehicle.steering, self.vehicle.throttle_brake, self.vehicle.speed
            )
        return

    def step(self, action: np.ndarray):

        if self.config["action_check"]:
            assert self.action_space.contains(action), "Input {} is not compatible with action space {}!".format(
                action, self.action_space
            )

        # prepare step
        if self.config["manual_control"] and self.use_render:
            action = self.controller.process_input()
        self.vehicle.prepare_step(action)
        self.traffic_manager.prepare_step()

        # ego vehicle/ traffic step
        for _ in range(self.config["decision_repeat"]):
            # traffic vehicles step
            self.traffic_manager.step(self.pg_world.pg_config["physics_world_step_size"])
            self.pg_world.step()

        # update states
        self.vehicle.update_state()
        self.traffic_manager.update_state(self.pg_world.physics_world)

        #  panda3d loop
        self.pg_world.taskMgr.step()

        # render before obtaining rgb observation
        if self.config["use_rgb"]:
            # when use rgb observation, the scene has to be drawn before using the camera data
            self.render()
        obs = self.observation.observe(self.vehicle)
        reward = self.reward(action)
        done_reward, done_info = self._done_episode()
        info = {
            "cost": float(0),
            "velocity": float(self.vehicle.speed),
            "steering": float(self.vehicle.steering),
            "acceleration": float(self.vehicle.throttle_brake),
            "step_reward": float(reward)
        }
        info.update(done_info)
        return obs, reward + done_reward, self.done, info

    def reset(self):
        self.lazy_init()  # it only works the first time when reset() is called to avoid the error when render
        self.done = False

        # clear world and traffic manager
        self.pg_world.clear_world()
        # select_map
        self.select_map()

        # reset main vehicle
        self.vehicle.reset(self.current_map, self.vehicle.born_place, 0.0)

        # generate new traffic according to the map
        self.traffic_manager.generate_traffic(
            self.pg_world, self.current_map, self.vehicle, self.config["traffic_density"]
        )
        o, *_ = self.step(np.array([0.0, 0.0]))
        return o

    def select_map(self):
        # remove map from world before adding
        if self.current_map is not None:
            self.current_map.remove_from_physics_world(self.pg_world.physics_world)
            self.current_map.remove_from_render_module()

        # create map
        self.current_seed = np.random.randint(self.start_seed, self.start_seed + self.env_num)
        if self.maps.get(self.current_seed, None) is None:
            map_config = self.config["map_config"]
            map_config.update({"seed": self.current_seed})
            new_map = Map(self.pg_world.worldNP, self.pg_world.physics_world, map_config)
            self.maps[self.current_seed] = new_map
            self.current_map = self.maps[self.current_seed]
        else:
            self.current_map = self.maps[self.current_seed]
            assert isinstance(self.current_map, Map), "map should be an instance of Map() class"
            self.current_map.re_generate(self.pg_world.worldNP, self.pg_world.physics_world)

    def reward(self, action):
        # Reward for moving forward in current lane
        current_lane = self.vehicle.lane
        long_last, _ = current_lane.local_coordinates(self.vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(self.vehicle.position)

        reward = 0.0
        lateral_factor = 1 - 2 * abs(lateral_now) / self.current_map.lane_width
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor

        # Penalty for frequent steering
        steering_change = abs(self.vehicle.last_current_action[0][0] - self.vehicle.last_current_action[1][0])
        steering_penalty = self.config["steering_penalty"] * steering_change * self.vehicle.speed / 20
        reward -= steering_penalty
        # Penalty for frequent acceleration / brake
        acceleration_penalty = self.config["acceleration_penalty"] * ((action[1])**2)
        reward -= acceleration_penalty

        # Penalty for waiting
        low_speed_penalty = 0
        if self.vehicle.speed < 1:
            low_speed_penalty = self.config["low_speed_penalty"]  # encourage car
        reward -= low_speed_penalty

        reward -= self.config["general_penalty"]

        reward += self.config["speed_reward"] * (self.vehicle.speed / self.vehicle.max_speed)

        return reward

    def _done_episode(self) -> (float, dict):
        reward_ = 0
        done_info = dict(crash=False, out_of_road=False, arrive_dest=False)
        long, lat = self.vehicle.routing_localization.final_lane.local_coordinates(self.vehicle.position)

        if self.vehicle.routing_localization.final_lane.length - 5 < long < self.vehicle.routing_localization.final_lane.length + 5 \
                and self.current_map.lane_width / 2 >= lat >= (
                0.5 - self.current_map.lane_num) * self.current_map.lane_width:
            self.done = True
            reward_ += self.config["success_reward"]
            print("arrive_dest")
            done_info["arrive_dest"] = True
        elif self.vehicle.crash:
            self.done = True
            reward_ -= self.config["crash_penalty"]
            print("crash")
            done_info["crash"] = True
        elif self.vehicle.out_of_road or self.vehicle.out_of_road:
            self.done = True
            reward_ -= self.config["out_of_road_penalty"]
            print("out_of_road")
            done_info["out_of_road"] = True

        return reward_, done_info

    def close(self):
        if self.pg_world is not None:
            self.vehicle.destroy(self.pg_world.physics_world)
            self.traffic_manager.destroy(self.pg_world.physics_world)

            del self.traffic_manager
            self.traffic_manager = None

            del self.control_camera
            self.control_camera = None

            del self.controller
            self.controller = None

            del self.vehicle
            self.vehicle = None

            self.pg_world.close_world()
            del self.pg_world
            self.pg_world = None
