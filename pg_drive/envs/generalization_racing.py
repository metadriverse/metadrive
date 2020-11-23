from pg_drive.pg_config.pg_config import PgConfig
import numpy as np
from pg_drive.world.chase_camera import ChaseCamera
import gym
from pg_drive.scene_creator.map import Map
from pg_drive.world.bt_world import BtWorld
from pg_drive.scene_creator.algorithm.BIG import BigGenerateMethod
from pg_drive.scene_creator.ego_vehicle.base_vehicle import BaseVehicle
from pg_drive.world.manual_controller import KeyboardController, JoystickController
from pg_drive.scene_manager.traffic_manager import TrafficManager, TrafficMode
from pg_drive.envs.observation_type import LidarStateObservation, ImageStateObservation


class GeneralizationRacing(gym.Env):
    def __init__(self, config: dict = None):
        self.config = self.default_config()
        if config:
            self.config.update(config)

        # set their value after vehicle created
        vehicle_config = BaseVehicle.get_vehicle_config(self.config["vehicle_config"])
        self.observation = LidarStateObservation(vehicle_config) if not self.config["use_rgb"] \
            else ImageStateObservation(vehicle_config)
        self.observation_space = self.observation.observation_space
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self.start_seed = self.config["start_seed"]
        self.env_num = self.config["environment_num"]
        self.map_config = self.config["map_config"]
        self.use_render = self.config["use_render"]
        bt_world_config = self.config["bt_world_config"]
        bt_world_config.update(
            {
                "use_render": self.use_render,
                "use_rgb": self.config["use_rgb"],
                "debug": self.config["debug"],
            }
        )
        self.bt_world_config = bt_world_config

        # lazy initialization, create the main vehicle in the lazy_init() func
        self.bullet_world = None
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
        if self.bullet_world is not None:
            return

        # init world
        self.bullet_world = BtWorld(self.bt_world_config)

        # init traffic manager
        self.traffic_manager = TrafficManager(self.config["traffic_mode"])

        # for manual_control and camera type
        if self.config["use_chase_camera"]:
            self.control_camera = ChaseCamera(1.8, 7, self.bullet_world)
        if self.config["manual_control"]:
            if self.config["controller"] == "keyboard":
                self.controller = KeyboardController()
            elif self.config["controller"] == "joystick":
                self.controller = JoystickController(self.bullet_world)
            else:
                raise ValueError("No such a controller type: {}".format(self.config["controller"]))

        # init vehicle
        v_config = self.config["vehicle_config"]
        self.vehicle = BaseVehicle(self.bullet_world, v_config)

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

            # ===== Traffic =====
            traffic_density=0.1,
            traffic_mode=TrafficMode.Add_once,

            # ===== Observation =====
            use_rgb=False,
            rgb_clip=True,
            vehicle_config=dict(),  # use default vehicle modules see more in BaseVehicle

            # ===== Road Network =====
            lane_width=3.5,
            lane_num=3,
            map_config=dict(type=BigGenerateMethod.BLOCK_NUM, config=3),

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
            steering_penalty=0.0,
            low_speed_penalty=0.1,
            driving_reward=1.0,
            general_penalty=0.0,
            speed_reward=0.0,

            # ===== Others =====
            bt_world_config=dict(),
            use_increment_steering=False,
        )
        return PgConfig(env_config)

    def render(self, mode='human', text: dict = None):
        assert self.use_render or self.config["use_rgb"], "render is off now, can not render"
        if self.control_camera is not None:
            self.control_camera.renew_camera_place(self.bullet_world.cam, self.vehicle)
        self.bullet_world.render_frame(text)
        if self.bullet_world.vehicle_panel is not None:
            self.bullet_world.vehicle_panel.renew_2d_car_para_visualization(
                self.vehicle.steering, self.vehicle.throttle_brake, self.vehicle.speed
            )
        return

    def step(self, action: np.ndarray):
        # prepare step
        if self.config["manual_control"] and self.use_render:
            action = self.controller.process_input()
        self.vehicle.prepare_step(action)
        self.traffic_manager.prepare_step()

        # ego vehicle/ traffic step
        for _ in range(self.config["decision_repeat"]):
            # traffic vehicles step
            self.traffic_manager.step(self.bullet_world.bt_config["bullet_world_step_size"])
            self.bullet_world.step()
            self.vehicle.collision_check()

        # update states
        self.vehicle.update_state()
        self.traffic_manager.update_state()

        #  panda3d loop
        self.bullet_world.taskMgr.step()

        # render before obtaining rgb observation
        if self.config["use_rgb"]:
            # when use rgb observation, the scene has to be drawn before using the camera data
            self.render()
        obs = self.observation.observe(self.vehicle)
        reward = self.reward(action)
        done_reward, done_info = self._done_episode()
        info = {"cost": 0, "velocity": self.vehicle.speed, "steering": self.vehicle.steering, "step_reward": reward}
        info.update(done_info)
        return obs, reward + done_reward, self.done, info

    def reset(self):
        self.lazy_init()  # it only works the first time when reset() is called to avoid the error when render
        self.done = False

        # clear world and traffic manager
        self.bullet_world.clear_world()
        # select_map
        self.select_map()

        # reset main vehicle
        self.vehicle.reset(self.current_map, self.vehicle.born_place, 0.0)

        # generate new traffic according to the map
        self.traffic_manager.generate_traffic(
            self.bullet_world, self.current_map, self.vehicle, self.current_seed, self.config["traffic_density"]
        )
        o, *_ = self.step(np.array([0.0, 0.0]))
        return o

    def select_map(self):
        # remove map from world before adding
        if self.current_map is not None:
            self.current_map.remove_from_physics_world(self.bullet_world.physics_world)
            self.current_map.remove_from_render_module()

        # create map
        self.current_seed = self.current_seed + 1 if self.current_seed < self.start_seed + self.env_num - 1 else self.start_seed
        if self.maps.get(self.current_seed, None) is None:
            new_map = Map(self.config["map_config"])
            new_map.big_generate(
                self.config["lane_width"], self.config["lane_num"], self.current_seed, self.bullet_world.worldNP,
                self.bullet_world.physics_world
            )
            self.maps[self.current_seed] = new_map
            self.current_map = self.maps[self.current_seed]
        else:
            self.current_map = self.maps[self.current_seed]
            assert isinstance(self.current_map, Map), "map should be an instance of Map() class"
            self.current_map.re_generate(self.bullet_world.worldNP, self.bullet_world.physics_world)

    def reward(self, action):
        # Reward for moving forward in current lane
        current_lane = self.vehicle.lane
        long_last, _ = current_lane.local_coordinates(self.vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(self.vehicle.position)

        reward = 0.0
        lateral_factor = 1 - 2 * abs(lateral_now) / self.config["lane_width"]
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor

        # print(f"[REWARD] Long now {long_now}, Long last {long_last}, reward {reward}")

        # Penalty for frequent steering
        steering_change = abs(self.vehicle.last_current_action[0][0] - self.vehicle.last_current_action[1][0])
        steering_penalty = self.config["steering_penalty"] * steering_change * self.vehicle.speed / 20
        reward -= steering_penalty
        # Penalty for frequent acceleration / brake
        acceleration_penalty = self.config["acceleration_penalty"] * ((action[1]) ** 2)
        reward -= acceleration_penalty

        # Penalty for waiting
        # reward -= 0.1
        low_speed_penalty = 0
        if self.vehicle.speed < 1:
            low_speed_penalty = self.config["low_speed_penalty"]  # encourage car
        reward -= low_speed_penalty

        reward -= self.config["general_penalty"]

        reward += self.config["speed_reward"] * (self.vehicle.speed / self.vehicle.max_speed)

        # print(f"reward {reward} steering {steering_penalty} acce {acceleration_penalty} low speed {low_speed_penalty}")

        # reward = reward - steering_penalty - acceleration_penalty - low_speed_penalty
        # # from city_drive.highway_env.utils import wrap_to_pi
        # lane_heading = current_right_lane.heading_at(long_now)
        # lane_direction = np.array([math.cos(lane_heading), math.sin(lane_heading)])
        # heading_dir = np.array([math.cos(self.vehicle.heading_theta), math.sin(self.vehicle.heading_theta)])
        #
        # heading_diff = wrap_to_pi(self.vehicle.heading_theta - lane_heading)
        #
        # speed_reward = self.vehicle.speed * (heading_dir[0] * lane_direction[0] + heading_dir[1] * lane_direction[1])

        # def f(a):
        #     return a / np.pi * 180
        #
        # def f2(v):
        #     return f(np.arctan2(v[1], v[0]))

        # if speed_reward < -1:
        #     print("Stop here")
        # raise ValueError("Wrong")

        # print("speed reward {:.5f} Longitude reward {:.5f}. Highway heading {:.5f}, Highway heading from vec {:.5f}, "
        #       "bullet heading {:.5f}, velocity dir {:.5f}, cur dir {:.5f}, ori {:.5f}. heading_diff {:.5f}".format(
        #     speed_reward,
        #     long_now - long_last,
        #     f(self.vehicle.heading_theta),
        #     f2(self.vehicle.heading),
        #     self.vehicle.chassis_np.getHpr()[0],
        #     # @f2(self.vehicle.forward_direction),
        #     # None,
        #     f2(self.vehicle.velocity_direction),
        #     f(current_right_lane.heading_at(long_now)),
        #     f2(self.vehicle.vehicle.get_forward_vector()),
        #     heading_diff
        #     # f(heading_dir)
        #     # long_now - long_last
        # ))

        # print("Longitude reward: ", long_now - long_last)

        return reward

    def _done_episode(self) -> (float, dict):
        reward_ = 0
        done_info = dict(crash=False, out_of_road=False, arrive_dest=False)
        long, lat = self.vehicle.routing_localization.final_lane.local_coordinates(self.vehicle.position)

        if self.vehicle.routing_localization.final_lane.length - 5 < long < self.vehicle.routing_localization.final_lane.length + 5 \
                and self.config["lane_width"] / 2 >= lat >= (0.5 - self.config["lane_num"]) * self.config["lane_width"]:
            self.done = True
            reward_ += self.config["success_reward"]
            print("arrive_dest")
            done_info["arrive_dest"] = True
        elif self.vehicle.crash:
            self.done = True
            reward_ -= self.config["crash_penalty"]
            done_info["crash"] = True
        elif self.vehicle.out_of_road or self.vehicle.out_of_road:
            self.done = True
            reward_ -= self.config["out_of_road_penalty"]
            done_info["out_of_road"] = True

        return reward_, done_info

    def close(self):
        if self.bullet_world is not None:
            self.vehicle.destroy(self.bullet_world.physics_world)
            self.traffic_manager.destroy(self.bullet_world.physics_world)
            self.bullet_world.close_world()
            del self.bullet_world
            self.bullet_world = None
