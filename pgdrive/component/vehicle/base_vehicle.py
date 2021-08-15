import math
from pgdrive.utils.math_utils import time_me
from collections import deque
from typing import Union, Optional

import gym
import numpy as np
import seaborn as sns
from panda3d.bullet import BulletVehicle, BulletBoxShape, ZUp
from panda3d.core import Material, Vec3, TransformState, LQuaternionf

from pgdrive.base_class.base_object import BaseObject
from pgdrive.component.lane.abs_lane import AbstractLane
from pgdrive.component.lane.circular_lane import CircularLane
from pgdrive.component.lane.straight_lane import StraightLane
from pgdrive.component.lane.waypoint_lane import WayPointLane
from pgdrive.component.map.base_map import BaseMap
from pgdrive.component.road.road import Road
from pgdrive.component.vehicle_module.depth_camera import DepthCamera
from pgdrive.component.vehicle_module.distance_detector import SideDetector, LaneLineDetector
from pgdrive.component.vehicle_module.lidar import Lidar
from pgdrive.component.vehicle_module.mini_map import MiniMap
from pgdrive.component.vehicle_module.rgb_camera import RGBCamera
from pgdrive.component.vehicle_module.navigation import Navigation
from pgdrive.constants import BodyName, CollisionGroup
from pgdrive.engine.asset_loader import AssetLoader
from pgdrive.engine.core.image_buffer import ImageBuffer
from pgdrive.engine.engine_utils import get_engine, engine_initialized
from pgdrive.engine.physics_node import BaseRigidBodyNode
from pgdrive.utils import Config, safe_clip_for_small_array, Vector
from pgdrive.utils import get_np_random
from pgdrive.utils.coordinates_shift import panda_position, pgdrive_position, panda_heading, pgdrive_heading
from pgdrive.utils.math_utils import get_vertical_vector, norm, clip
from pgdrive.utils.scene_utils import ray_localization
from pgdrive.utils.scene_utils import rect_region_detection
from pgdrive.utils.space import ParameterSpace, Parameter, VehicleParameterSpace


class BaseVehicleState:
    def __init__(self):
        self.crash_vehicle = False
        self.crash_object = False
        self.crash_sidewalk = False
        self.crash_building = False

        # lane line detection
        self.on_yellow_continuous_line = False
        self.on_white_continuous_line = False
        self.on_broken_line = False

        # contact results, a set containing objects type name for rendering
        self.contact_results = None

    def init_state_info(self):
        """
        Call this before reset()/step()
        """
        self.crash_vehicle = False
        self.crash_object = False
        self.crash_sidewalk = False
        self.crash_building = False

        self.on_yellow_continuous_line = False
        self.on_white_continuous_line = False
        self.on_broken_line = False

        # contact results
        self.contact_results = None


class BaseVehicle(BaseObject, BaseVehicleState):
    """
    Vehicle chassis and its wheels index
                    0       1
                    II-----II
                        |
                        |  <---chassis
                        |
                    II-----II
                    2       3
    """
    MODEL = None
    PARAMETER_SPACE = ParameterSpace(
        VehicleParameterSpace.BASE_VEHICLE
    )  # it will not sample config from parameter space
    COLLISION_MASK = CollisionGroup.Vehicle
    STEERING_INCREMENT = 0.05

    LENGTH = 4.51
    WIDTH = 1.852
    HEIGHT = 1.19

    # for random color choosing
    MATERIAL_COLOR_COEFF = 10  # to resist other factors, since other setting may make color dark
    MATERIAL_METAL_COEFF = 1  # 0-1
    MATERIAL_ROUGHNESS = 0.8  # smaller to make it more smooth, and reflect more light
    MATERIAL_SHININESS = 1  # 0-128 smaller to make it more smooth, and reflect more light
    MATERIAL_SPECULAR_COLOR = (3, 3, 3, 3)

    def __init__(
        self,
        vehicle_config: Union[dict, Config] = None,
        name: str = None,
        random_seed=None,
    ):
        """
        This Vehicle Config is different from self.get_config(), and it is used to define which modules to use, and
        module parameters. And self.physics_config defines the physics feature of vehicles, such as length/width
        :param vehicle_config: mostly, vehicle module config
        :param random_seed: int
        """
        # check
        assert vehicle_config is not None, "Please specify the vehicle config."
        assert engine_initialized(), "Please make sure game engine is successfully initialized!"

        # NOTE: it is the game engine, not vehicle drivetrain
        self.engine = get_engine()
        BaseObject.__init__(self, name, random_seed, self.engine.global_config["vehicle_config"])
        BaseVehicleState.__init__(self)
        self.update_config(vehicle_config)
        am_i_the_special_one = self.config["am_i_the_special_one"]

        # build vehicle physics model
        vehicle_chassis = self._create_vehicle_chassis()
        self.add_body(vehicle_chassis.getChassis())
        self.system = vehicle_chassis
        self.chassis = self.origin
        self.wheels = self._create_wheel()

        # powertrain config
        self.increment_steering = self.config["increment_steering"]
        self.enable_reverse = self.config["enable_reverse"]
        self.max_speed = self.config["max_speed"]
        self.max_steering = self.config["max_steering"]

        # visualization
        color = sns.color_palette("colorblind")
        idx = get_np_random().randint(len(color))
        rand_c = color[idx]
        if am_i_the_special_one:
            rand_c = color[2]  # A pretty green
        self.top_down_color = (rand_c[0] * 255, rand_c[1] * 255, rand_c[2] * 255)
        self.panda_color = rand_c
        self._add_visualization()

        # modules, get observation by using these modules
        self.lane: Optional[AbstractLane] = None
        self.lane_index = None
        self.navigation: Optional[Navigation] = None
        self.lidar: Optional[Lidar] = None  # detect surrounding vehicles
        self.side_detector: Optional[SideDetector] = None  # detect road side
        self.lane_line_detector: Optional[LaneLineDetector] = None  # detect nearest lane lines
        self.image_sensors = {}

        # state info
        self.throttle_brake = 0.0
        self.steering = 0
        self.last_current_action = deque([(0.0, 0.0), (0.0, 0.0)], maxlen=2)
        self.last_position = (0, 0)
        self.last_heading_dir = self.heading
        self.dist_to_left_side = None
        self.dist_to_right_side = None

        # step info
        self.out_of_route = None
        self.on_lane = None
        self.spawn_place = (0, 0)
        self._init_step_info()

        # others
        self._add_modules_for_vehicle()
        self.takeover = False
        self._expert_takeover = False
        self.energy_consumption = 0
        self.action_space = self.get_action_space_before_init(extra_action_dim=self.config["extra_action_dim"])

        # overtake_stat
        self.front_vehicles = set()
        self.back_vehicles = set()

        if self.engine.current_map is not None:
            self.reset()

    def _add_modules_for_vehicle(self, ):
        config = self.config

        # add routing module
        self.add_routing_localization(config["show_navi_mark"])  # default added

        # add distance detector/lidar
        self.side_detector = SideDetector(
            config["side_detector"]["num_lasers"], config["side_detector"]["distance"], config["show_side_detector"]
        )

        self.lane_line_detector = LaneLineDetector(
            config["lane_line_detector"]["num_lasers"], config["lane_line_detector"]["distance"],
            config["show_lane_line_detector"]
        )

        self.lidar = Lidar(config["lidar"]["num_lasers"], config["lidar"]["distance"], config["show_lidar"])

        # vision modules
        self.add_image_sensor("rgb_camera", RGBCamera())
        self.add_image_sensor("mini_map", MiniMap())
        self.add_image_sensor("depth_camera", DepthCamera())

    def _init_step_info(self):
        # done info will be initialized every frame
        self.init_state_info()
        self.out_of_route = False  # re-route is required if is false
        self.on_lane = True  # on lane surface or not

    def _preprocess_action(self, action):
        if self.config["action_check"]:
            assert self.action_space.contains(action), "Input {} is not compatible with action space {}!".format(
                action, self.action_space
            )

        # protect agent from nan error
        action = safe_clip_for_small_array(action, min_val=self.action_space.low[0], max_val=self.action_space.high[0])
        return action, {'raw_action': (action[0], action[1])}

    def before_step(self, action):
        """
        Save info and make decision before action
        """
        # init step info to store info before each step
        self._init_step_info()
        action, step_info = self._preprocess_action(action)

        self.last_position = self.position
        self.last_heading_dir = self.heading
        self.last_current_action.append(action)  # the real step of physics world is implemented in taskMgr.step()
        if self.increment_steering:
            self._set_incremental_action(action)
        else:
            self._set_action(action)
        return step_info

    def after_step(self):
        if self.navigation is not None:
            self.lane, self.lane_index, = self.navigation.update_localization(self)
        self._state_check()
        self.update_dist_to_left_right()
        step_energy, episode_energy = self._update_energy_consumption()
        self.out_of_route = self._out_of_route()
        step_info = self._update_overtake_stat()
        step_info.update(
            {
                "velocity": float(self.speed),
                "steering": float(self.steering),
                "acceleration": float(self.throttle_brake),
                "step_energy": step_energy,
                "episode_energy": episode_energy
            }
        )
        return step_info

    def _out_of_route(self):
        left, right = self._dist_to_route_left_right()
        return True if right < 0 or left < 0 else False

    def _update_energy_consumption(self):
        """
        The calculation method is from
        https://www.researchgate.net/publication/262182035_Reduction_of_Fuel_Consumption_and_Exhaust_Pollutant_Using_Intelligent_Transport_System
        default: 3rd gear, try to use ae^bx to fit it, dp: (90, 8), (130, 12)
        :return: None
        """
        distance = norm(*(self.last_position - self.position)) / 1000  # km
        step_energy = 3.25 * math.pow(np.e, 0.01 * self.speed) * distance / 100
        # step_energy is in Liter, we return mL
        step_energy = step_energy * 1000
        self.energy_consumption += step_energy  # L/100 km
        return step_energy, self.energy_consumption

    def reset(self, random_seed=None, vehicle_config=None, pos: np.ndarray = None, heading: float = 0.0):
        """
        pos is a 2-d array, and heading is a float (unit degree)
        if pos is not None, vehicle will be reset to the position
        else, vehicle will be reset to spawn place
        """
        if random_seed is not None:
            self.seed(random_seed)
        if vehicle_config is not None:
            self.update_config(vehicle_config)
        map = self.engine.current_map
        if pos is None:
            lane = map.road_network.get_lane(self.config["spawn_lane_index"])
            pos = lane.position(self.config["spawn_longitude"], self.config["spawn_lateral"])
            heading = np.rad2deg(lane.heading_at(self.config["spawn_longitude"]))
            self.spawn_place = pos
        heading = -np.deg2rad(heading) - np.pi / 2
        self.set_static(False)
        self.origin.setPos(panda_position(Vec3(*pos, self.HEIGHT / 2 + 1)))
        self.origin.setQuat(LQuaternionf(math.cos(heading / 2), 0, 0, math.sin(heading / 2)))
        self.update_map_info(map)
        self.body.clearForces()
        self.body.setLinearVelocity(Vec3(0, 0, 0))
        self.body.setAngularVelocity(Vec3(0, 0, 0))
        self.system.resetSuspension()
        # np.testing.assert_almost_equal(self.position, pos, decimal=4)

        # done info
        self._init_step_info()

        # other info
        self.throttle_brake = 0.0
        self.steering = 0
        self.last_current_action = deque([(0.0, 0.0), (0.0, 0.0)], maxlen=2)
        self.last_position = self.spawn_place
        self.last_heading_dir = self.heading

        self.update_dist_to_left_right()
        self.takeover = False
        self.energy_consumption = 0

        # overtake_stat
        self.front_vehicles = set()
        self.back_vehicles = set()

        assert self.navigation

    """------------------------------------------- act -------------------------------------------------"""

    def _set_action(self, action):
        steering = action[0]
        self.throttle_brake = action[1]
        self.steering = steering
        self.system.setSteeringValue(self.steering * self.max_steering, 0)
        self.system.setSteeringValue(self.steering * self.max_steering, 1)
        self._apply_throttle_brake(action[1])

    def _set_incremental_action(self, action: np.ndarray):
        self.throttle_brake = action[1]
        self.steering += action[0] * self.STEERING_INCREMENT
        self.steering = clip(self.steering, -1, 1)
        steering = self.steering * self.max_steering
        self.system.setSteeringValue(steering, 0)
        self.system.setSteeringValue(steering, 1)
        self._apply_throttle_brake(action[1])

    def _apply_throttle_brake(self, throttle_brake):
        max_engine_force = self.config["max_engine_force"]
        max_brake_force = self.config["max_brake_force"]
        for wheel_index in range(4):
            if throttle_brake >= 0:
                self.system.setBrake(2.0, wheel_index)
                if self.speed > self.max_speed:
                    self.system.applyEngineForce(0.0, wheel_index)
                else:
                    self.system.applyEngineForce(max_engine_force * throttle_brake, wheel_index)
            else:
                if self.enable_reverse:
                    self.system.applyEngineForce(max_engine_force * throttle_brake, wheel_index)
                    self.system.setBrake(0, wheel_index)
                else:
                    self.system.applyEngineForce(0.0, wheel_index)
                    self.system.setBrake(abs(throttle_brake) * max_brake_force, wheel_index)

    """---------------------------------------- vehicle info ----------------------------------------------"""

    def update_dist_to_left_right(self):
        self.dist_to_left_side, self.dist_to_right_side = self._dist_to_route_left_right()

    def _dist_to_route_left_right(self):
        current_reference_lane = self.navigation.current_ref_lanes[0]
        _, lateral_to_reference = current_reference_lane.local_coordinates(self.position)
        lateral_to_left = lateral_to_reference + self.navigation.get_current_lane_width() / 2
        lateral_to_right = self.navigation.get_current_lateral_range(self.position, self.engine) - lateral_to_left
        return lateral_to_left, lateral_to_right

    @property
    def position(self):
        return pgdrive_position(self.origin.getPos())

    @property
    def speed(self):
        """
        km/h
        """
        velocity = self.body.get_linear_velocity()
        speed = norm(velocity[0], velocity[1]) * 3.6
        return clip(speed, 0.0, 100000.0)

    @property
    def heading(self):
        real_heading = self.heading_theta
        # heading = np.array([math.cos(real_heading), math.sin(real_heading)])
        heading = Vector((math.cos(real_heading), math.sin(real_heading)))
        return heading

    @property
    def heading_theta(self):
        """
        Get the heading theta of vehicle, unit [rad]
        :return:  heading in rad
        """
        return (pgdrive_heading(self.origin.getH()) - 90) / 180 * math.pi

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.velocity_direction

    @property
    def velocity_direction(self):
        direction = self.system.getForwardVector()
        return np.asarray([direction[0], -direction[1]])

    @property
    def current_road(self):
        return Road(*self.lane_index[0:-1])

    """---------------------------------------- some math tool ----------------------------------------------"""

    def heading_diff(self, target_lane):
        lateral = None
        if isinstance(target_lane, StraightLane):
            lateral = np.asarray(get_vertical_vector(target_lane.end - target_lane.start)[1])
        elif isinstance(target_lane, CircularLane):
            if target_lane.direction == -1:
                lateral = self.position - target_lane.center
            else:
                lateral = target_lane.center - self.position
        elif isinstance(target_lane, WayPointLane):
            lane_segment = target_lane.segment(target_lane.local_coordinates(self.position)[0])
            lateral = lane_segment["lateral_direction"]

        lateral_norm = norm(lateral[0], lateral[1])
        forward_direction = self.heading
        # print(f"Old forward direction: {self.forward_direction}, new heading {self.heading}")
        forward_direction_norm = norm(forward_direction[0], forward_direction[1])
        if not lateral_norm * forward_direction_norm:
            return 0
        cos = (
            (forward_direction[0] * lateral[0] + forward_direction[1] * lateral[1]) /
            (lateral_norm * forward_direction_norm)
        )
        # return cos
        # Normalize to 0, 1
        return clip(cos, -1.0, 1.0) / 2 + 0.5

    def projection(self, vector):
        # Projected to the heading of vehicle
        # forward_vector = self.vehicle.get_forward_vector()
        # forward_old = (forward_vector[0], -forward_vector[1])

        forward = self.heading

        # print(f"[projection] Old forward {forward_old}, new heading {forward}")

        norm_velocity = norm(forward[0], forward[1]) + 1e-6
        project_on_heading = (vector[0] * forward[0] + vector[1] * forward[1]) / norm_velocity

        side_direction = get_vertical_vector(forward)[1]
        side_norm = norm(side_direction[0], side_direction[1]) + 1e-6
        project_on_side = (vector[0] * side_direction[0] + vector[1] * side_direction[1]) / side_norm
        return project_on_heading, project_on_side

    def lane_distance_to(self, vehicle, lane: AbstractLane = None) -> float:
        assert self.navigation is not None, "a routing and localization module should be added " \
                                            "to interact with other vehicles"
        if not vehicle:
            return np.nan
        if not lane:
            lane = self.lane
        return lane.local_coordinates(vehicle.position)[0] - lane.local_coordinates(self.position)[0]

    """-------------------------------------- for vehicle making ------------------------------------------"""

    def _create_vehicle_chassis(self):
        para = self.get_config()

        self.LENGTH = type(self).LENGTH  # or self.config["vehicle_length"]
        self.WIDTH = type(self).WIDTH  # or self.config["vehicle_width"]
        self.HEIGHT = type(self).HEIGHT  # or self.config[Parameter.vehicle_height]

        chassis = BaseRigidBodyNode(self.name, BodyName.Vehicle)
        chassis.setIntoCollideMask(CollisionGroup.Vehicle)
        chassis_shape = BulletBoxShape(Vec3(self.WIDTH / 2, self.LENGTH / 2, self.HEIGHT / 2))
        ts = TransformState.makePos(Vec3(0, 0, self.HEIGHT / 2))
        chassis.addShape(chassis_shape, ts)
        chassis.setMass(para[Parameter.mass])
        chassis.setDeactivationEnabled(False)
        chassis.notifyCollisions(True)  # advance collision check, do callback in pg_collision_callback

        physics_world = get_engine().physics_world
        vehicle_chassis = BulletVehicle(physics_world.dynamic_world, chassis)
        vehicle_chassis.setCoordinateSystem(ZUp)
        self.dynamic_nodes.append(vehicle_chassis)
        return vehicle_chassis

    def _add_visualization(self):
        if self.render:

            if self.MODEL is None:
                model_path = 'models/ferra/scene.gltf'
                self.MODEL = self.loader.loadModel(AssetLoader.file_path(model_path))
                self.MODEL.setZ(-self.config[Parameter.tire_radius] - 0.2)
                self.MODEL.set_scale(1)

            self.MODEL.instanceTo(self.origin)
            if self.config["random_color"]:
                material = Material()
                material.setBaseColor(
                    (
                        self.panda_color[0] * self.MATERIAL_COLOR_COEFF,
                        self.panda_color[1] * self.MATERIAL_COLOR_COEFF,
                        self.panda_color[2] * self.MATERIAL_COLOR_COEFF, 0.2
                    )
                )
                material.setMetallic(self.MATERIAL_METAL_COEFF)
                material.setSpecular(self.MATERIAL_SPECULAR_COLOR)
                material.setRefractiveIndex(1.5)
                material.setRoughness(self.MATERIAL_ROUGHNESS)
                material.setShininess(self.MATERIAL_SHININESS)
                material.setTwoside(False)
                self.origin.setMaterial(material, True)

    def _create_wheel(self):
        para = self.get_config()
        f_l = para[Parameter.front_tire_longitude]
        r_l = -para[Parameter.rear_tire_longitude]
        lateral = para[Parameter.tire_lateral]
        axis_height = para[Parameter.tire_radius] - 0.2  # 0.2 suspension length
        radius = para[Parameter.tire_radius]
        wheels = []
        for k, pos in enumerate([Vec3(lateral, f_l, axis_height), Vec3(-lateral, f_l, axis_height),
                                 Vec3(lateral, r_l, axis_height), Vec3(-lateral, r_l, axis_height)]):
            wheel = self._add_wheel(pos, radius, True if k < 2 else False, True if k == 0 or k == 2 else False)
            wheels.append(wheel)
        return wheels

    def _add_wheel(self, pos: Vec3, radius: float, front: bool, left):
        wheel_np = self.origin.attachNewNode("wheel")
        if self.render:
            # TODO something wrong with the wheel render
            model_path = 'models/yugo/yugotireR.egg' if left else 'models/yugo/yugotireL.egg'
            wheel_model = self.loader.loadModel(AssetLoader.file_path(model_path))
            wheel_model.reparentTo(wheel_np)
            wheel_model.set_scale(1.4, radius / 0.25, radius / 0.25)
        wheel = self.system.create_wheel()
        wheel.setNode(wheel_np.node())
        wheel.setChassisConnectionPointCs(pos)
        wheel.setFrontWheel(front)
        wheel.setWheelDirectionCs(Vec3(0, 0, -1))
        wheel.setWheelAxleCs(Vec3(1, 0, 0))

        wheel.setWheelRadius(radius)
        wheel.setMaxSuspensionTravelCm(40)
        wheel.setSuspensionStiffness(30)
        wheel.setWheelsDampingRelaxation(4.8)
        wheel.setWheelsDampingCompression(1.2)
        wheel.setFrictionSlip(self.config["wheel_friction"])
        wheel.setRollInfluence(1.5)
        return wheel

    def add_image_sensor(self, name: str, sensor: ImageBuffer):
        self.image_sensors[name] = sensor

    def add_routing_localization(self, show_navi_mark: bool = False):
        config = self.config
        self.navigation = Navigation(
            self.engine,
            show_navi_mark=show_navi_mark,
            random_navi_mark_color=config["random_navi_mark_color"],
            show_dest_mark=config["show_dest_mark"],
            show_line_to_dest=config["show_line_to_dest"]
        )

    def update_map_info(self, map):
        """
        Update map info after reset()
        :param map: new map
        :return: None
        """
        possible_lanes = ray_localization(self.heading, self.spawn_place, self.engine, return_all_result=True)
        possible_lane_indexes = [lane_index for lane, lane_index, dist in possible_lanes]
        try:
            idx = possible_lane_indexes.index(self.config["spawn_lane_index"])
        except ValueError:
            lane, new_l_index = possible_lanes[0][:-1]
        else:
            lane, new_l_index = possible_lanes[idx][:-1]
        dest = self.config["destination_node"]
        self.navigation.update(map, current_lane_index=new_l_index, final_road_node=dest if dest is not None else None)
        assert lane is not None, "spawn place is not on road!"
        self.lane_index = new_l_index
        self.lane = lane

    def _state_check(self):
        """
        Check States and filter to update info
        """
        result_1 = self.engine.physics_world.static_world.contactTest(self.chassis.node(), True)
        result_2 = self.engine.physics_world.dynamic_world.contactTest(self.chassis.node(), True)
        contacts = set()
        for contact in result_1.getContacts() + result_2.getContacts():
            node0 = contact.getNode0()
            node1 = contact.getNode1()
            name = [node0.getName(), node1.getName()]
            name.remove(BodyName.Vehicle)
            if name[0] == BodyName.White_continuous_line:
                self.on_white_continuous_line = True
            elif name[0] == BodyName.Yellow_continuous_line:
                self.on_yellow_continuous_line = True
            elif name[0] == BodyName.Broken_line:
                self.on_broken_line = True
            else:
                # didn't add
                continue
            contacts.add(name[0])
        # side walk detect
        res = rect_region_detection(
            self.engine, self.position, np.rad2deg(self.heading_theta), self.LENGTH, self.WIDTH, CollisionGroup.Sidewalk
        )
        if res.hasHit():
            self.crash_sidewalk = True
            contacts.add(BodyName.Sidewalk)
        self.contact_results = contacts

    def destroy(self):
        super(BaseVehicle, self).destroy()

        self.navigation.destroy()
        self.navigation = None

        self.side_detector.destroy()
        self.lane_line_detector.destroy()
        self.lidar.destroy()
        self.side_detector = None
        self.lane_line_detector = None
        self.lidar = None
        if len(self.image_sensors) != 0:
            for sensor in self.image_sensors.values():
                sensor.destroy()
        self.image_sensors = None
        self.engine = None

    def set_position(self, position, height=0.4):
        """
        Should only be called when restore traffic from episode data
        :param position: 2d array or list
        :return: None
        """
        self.origin.setPos(panda_position(position, height))

    def set_heading(self, heading_theta) -> None:
        """
        Should only be called when restore traffic from episode data
        :param heading_theta: float in rad
        :return: None
        """
        self.origin.setH((panda_heading(heading_theta) * 180 / np.pi) - 90)

    def get_state(self):
        final_road = self.navigation.final_road
        return {
            "heading": self.heading_theta,
            "position": self.position.tolist(),
            "done": self.crash_vehicle or self.out_of_route or self.crash_sidewalk or not self.on_lane,
            "speed": self.speed,
            "spawn_road": self.config["spawn_lane_index"][:-1],
            "destination": (final_road.start_node, final_road.end_node)
        }

    def set_state(self, state: dict):
        self.set_heading(state["heading"])
        self.set_position(state["position"])
        self._replay_done = state["done"]
        self.set_position(state["position"], height=0.28)

    def _update_overtake_stat(self):
        if self.config["overtake_stat"] and self.lidar.available:
            surrounding_vs = self.lidar.get_surrounding_vehicles()
            routing = self.navigation
            ckpt_idx = routing._target_checkpoints_index
            for surrounding_v in surrounding_vs:
                if surrounding_v.lane_index[:-1] == (routing.checkpoints[ckpt_idx[0]], routing.checkpoints[ckpt_idx[1]
                                                                                                           ]):
                    if self.lane.local_coordinates(self.position)[0] - \
                            self.lane.local_coordinates(surrounding_v.position)[0] < 0:
                        self.front_vehicles.add(surrounding_v)
                        if surrounding_v in self.back_vehicles:
                            self.back_vehicles.remove(surrounding_v)
                    else:
                        self.back_vehicles.add(surrounding_v)
        return {"overtake_vehicle_num": self.get_overtake_num()}

    def get_overtake_num(self):
        return len(self.front_vehicles.intersection(self.back_vehicles))

    @classmethod
    def get_action_space_before_init(cls, extra_action_dim: int = 0):
        return gym.spaces.Box(-1.0, 1.0, shape=(2 + extra_action_dim, ), dtype=np.float32)

    def __del__(self):
        super(BaseVehicle, self).__del__()
        self.engine = None
        self.lidar = None
        self.mini_map = None
        self.rgb_camera = None
        self.navigation = None
        self.wheels = None

    @property
    def arrive_destination(self):
        long, lat = self.navigation.final_lane.local_coordinates(self.position)
        flag = (self.navigation.final_lane.length - 5 < long < self.navigation.final_lane.length + 5) and (
            self.navigation.get_current_lane_width() / 2 >= lat >=
            (0.5 - self.navigation.get_current_lane_num()) * self.navigation.get_current_lane_width()
        )
        return flag

    def set_static(self, flag):
        self.body.setStatic(flag)

    @property
    def reference_lanes(self):
        return self.navigation.current_ref_lanes

    def set_wheel_friction(self, new_friction):
        raise DeprecationWarning("Bug exists here")
        for wheel in self.wheels:
            wheel.setFrictionSlip(new_friction)

    @property
    def overspeed(self):
        return True if self.lane.speed_limit < self.speed else False

    @property
    def replay_done(self):
        return self._replay_done if hasattr(self, "_replay_done") else (
            self.crash_building or self.crash_vehicle or
            # self.on_white_continuous_line or
            self.on_yellow_continuous_line
        )

    @property
    def current_action(self):
        return self.last_current_action[-1]

    @property
    def last_current(self):
        return self.last_current_action[0]
