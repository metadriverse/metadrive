import math
from collections import deque
from typing import Union, Optional

import gym
import numpy as np
import seaborn as sns
from metadrive.base_class.base_object import BaseObject
from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.lane.circular_lane import CircularLane
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.lane.waypoint_lane import WayPointLane
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.component.vehicle_module.depth_camera import DepthCamera
from metadrive.component.vehicle_module.distance_detector import SideDetector, LaneLineDetector
from metadrive.component.vehicle_module.lidar import Lidar
from metadrive.component.vehicle_module.mini_map import MiniMap
from metadrive.component.vehicle_module.rgb_camera import RGBCamera
from metadrive.component.vehicle_navigation_module.edge_network_navigation import EdgeNetworkNavigation
from metadrive.component.vehicle_navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.constants import BodyName, CollisionGroup
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.core.image_buffer import ImageBuffer
from metadrive.engine.engine_utils import get_engine, engine_initialized
from metadrive.engine.physics_node import BaseRigidBodyNode
from metadrive.utils import Config, safe_clip_for_small_array
from metadrive.utils.coordinates_shift import panda_heading, metadrive_heading
from metadrive.utils.math_utils import get_vertical_vector, norm, clip
from metadrive.utils.math_utils import wrap_to_pi
from metadrive.utils.scene_utils import ray_localization
from metadrive.utils.scene_utils import rect_region_detection
from metadrive.utils.space import VehicleParameterSpace, ParameterSpace
from panda3d.bullet import BulletVehicle, BulletBoxShape, ZUp
from panda3d.core import Material, Vec3, TransformState, LQuaternionf


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
        self.contact_results = set()


class BaseVehicle(BaseObject, BaseVehicleState):
    """
    Vehicle chassis and its wheels index
                    0       1
                    II-----II
                        |
                        |  <---chassis/wheelbase
                        |
                    II-----II
                    2       3
    """
    COLLISION_MASK = CollisionGroup.Vehicle
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.BASE_VEHICLE)
    MAX_LENGTH = 10
    MAX_WIDTH = 2.5
    MAX_STEERING = 60

    LENGTH = None
    WIDTH = None
    HEIGHT = None
    TIRE_RADIUS = None
    LATERAL_TIRE_TO_CENTER = None
    TIRE_WIDTH = 0.4
    FRONT_WHEELBASE = None
    REAR_WHEELBASE = None
    MASS = None
    CHASSIS_TO_WHEEL_AXIS = 0.2
    SUSPENSION_LENGTH = 15
    SUSPENSION_STIFFNESS = 40

    # for random color choosing
    MATERIAL_COLOR_COEFF = 10  # to resist other factors, since other setting may make color dark
    MATERIAL_METAL_COEFF = 1  # 0-1
    MATERIAL_ROUGHNESS = 0.8  # smaller to make it more smooth, and reflect more light
    MATERIAL_SHININESS = 1  # 0-128 smaller to make it more smooth, and reflect more light
    MATERIAL_SPECULAR_COLOR = (3, 3, 3, 3)

    # control
    STEERING_INCREMENT = 0.05

    # save memory, load model once
    model_collection = {}
    path = None

    def __init__(
        self,
        vehicle_config: Union[dict, Config] = None,
        name: str = None,
        random_seed=None,
        position=None,
        heading=None
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
        use_special_color = self.config["use_special_color"]

        # build vehicle physics model
        vehicle_chassis = self._create_vehicle_chassis()
        self.add_body(vehicle_chassis.getChassis())
        self.system = vehicle_chassis
        self.chassis = self.origin
        self.wheels = self._create_wheel()

        # powertrain config
        self.increment_steering = self.config["increment_steering"]
        self.enable_reverse = self.config["enable_reverse"]
        self.max_steering = self.config["max_steering"]

        # visualization
        self._use_special_color = use_special_color
        self._add_visualization()

        # modules, get observation by using these modules
        self.navigation: Optional[NodeNetworkNavigation] = None
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
        self.expert_takeover = False
        self.energy_consumption = 0
        self.action_space = self.get_action_space_before_init(extra_action_dim=self.config["extra_action_dim"])
        self.break_down = False

        # overtake_stat
        self.front_vehicles = set()
        self.back_vehicles = set()

        if self.engine.current_map is not None:
            self.reset(position=position, heading=heading)

    def _add_modules_for_vehicle(self, ):
        config = self.config

        # add routing module
        self.add_navigation()  # default added

        # add distance detector/lidar
        self.side_detector = SideDetector(
            config["side_detector"]["num_lasers"], config["side_detector"]["distance"],
            self.engine.global_config["vehicle_config"]["show_side_detector"]
        )

        self.lane_line_detector = LaneLineDetector(
            config["lane_line_detector"]["num_lasers"], config["lane_line_detector"]["distance"],
            self.engine.global_config["vehicle_config"]["show_lane_line_detector"]
        )

        self.lidar = Lidar(
            config["lidar"]["num_lasers"], config["lidar"]["distance"],
            self.engine.global_config["vehicle_config"]["show_lidar"]
        )

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
        action = safe_clip_for_small_array(action, -1, 1)
        if self.config["action_check"]:
            assert self.action_space.contains(action), "Input {} is not compatible with action space {}!".format(
                action, self.action_space
            )
        return action, {'raw_action': (action[0], action[1])}

    def before_step(self, action=None):
        """
        Save info and make decision before action
        """
        # init step info to store info before each step
        if action is None:
            action = [0, 0]
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
            self.navigation.update_localization(self)
        self._state_check()
        self.update_dist_to_left_right()
        step_energy, episode_energy = self._update_energy_consumption()
        self.out_of_route = self._out_of_route()
        step_info = self._update_overtake_stat()
        my_policy = self.engine.get_policy(self.name)
        step_info.update(
            {
                "velocity": float(self.speed),
                "steering": float(self.steering),
                "acceleration": float(self.throttle_brake),
                "step_energy": step_energy,
                "episode_energy": episode_energy,
                "policy": my_policy.name if my_policy is not None else my_policy
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

    def reset(
        self,
        random_seed=None,
        vehicle_config=None,
        position: np.ndarray = None,
        heading: float = 0.0,
        *args,
        **kwargs
    ):
        """
        pos is a 2-d array, and heading is a float (unit degree)
        if pos is not None, vehicle will be reset to the position
        else, vehicle will be reset to spawn place
        """
        if random_seed is not None:
            self.seed(random_seed)
            self.sample_parameters()
        if vehicle_config is not None:
            self.update_config(vehicle_config)
        map = self.engine.current_map

        if position is not None:
            # Highest priority
            pass
        elif self.config["spawn_position_heading"] is None:
            # spawn_lane_index has second priority
            lane = map.road_network.get_lane(self.config["spawn_lane_index"])
            position = lane.position(self.config["spawn_longitude"], self.config["spawn_lateral"])
            heading = np.rad2deg(lane.heading_theta_at(self.config["spawn_longitude"]))
        else:
            assert self.config["spawn_position_heading"] is not None, "At least setting one initialization method"
            position = self.config["spawn_position_heading"][0]
            heading = self.config["spawn_position_heading"][1]

        self.spawn_place = position
        heading = -np.deg2rad(heading) - np.pi / 2
        self.set_static(False)
        self.set_position(position, self.HEIGHT / 2 + 1)
        self.origin.setQuat(LQuaternionf(math.cos(heading / 2), 0, 0, math.sin(heading / 2)))
        self.update_map_info(map)
        self.body.clearForces()
        self.body.setLinearVelocity(Vec3(0, 0, 0))
        self.body.setAngularVelocity(Vec3(0, 0, 0))
        self.system.resetSuspension()
        self._apply_throttle_brake(0.0)
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
        self.expert_takeover = False
        if self.config["need_navigation"]:
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
        # TODO
        if self.navigation is None:
            return 0, 0
        current_reference_lane = self.navigation.current_ref_lanes[0]
        _, lateral_to_reference = current_reference_lane.local_coordinates(self.position)
        lateral_to_left = lateral_to_reference + self.navigation.get_current_lane_width() / 2
        lateral_to_right = self.navigation.get_current_lateral_range(self.position, self.engine) - lateral_to_left
        return lateral_to_left, lateral_to_right

    @property
    def heading_theta(self):
        """
        Get the heading theta of vehicle, unit [rad]
        :return:  heading in rad
        """
        return (metadrive_heading(self.origin.getH()) - 90) / 180 * math.pi

    @property
    def speed(self):
        """
        km/h
        """
        velocity = self.body.get_linear_velocity()
        speed = norm(velocity[0], velocity[1]) * 3.6
        return clip(speed, 0.0, 100000.0)

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.velocity_direction

    @property
    def velocity_direction(self):
        direction = self.system.getForwardVector()
        return np.asarray([direction[0], -direction[1]])

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
            lateral = target_lane.lateral_direction(target_lane.local_coordinates(self.position)[0])

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
        self.LENGTH = type(self).LENGTH
        self.WIDTH = type(self).WIDTH
        self.HEIGHT = type(self).HEIGHT
        assert self.LENGTH < BaseVehicle.MAX_LENGTH, "Vehicle is too large!"
        assert self.WIDTH < BaseVehicle.MAX_WIDTH, "Vehicle is too large!"

        chassis = BaseRigidBodyNode(self.name, BodyName.Vehicle)
        chassis.setIntoCollideMask(CollisionGroup.Vehicle)
        chassis_shape = BulletBoxShape(Vec3(self.WIDTH / 2, self.LENGTH / 2, self.HEIGHT / 2))
        ts = TransformState.makePos(Vec3(0, 0, self.HEIGHT / 2))
        chassis.addShape(chassis_shape, ts)
        chassis.setDeactivationEnabled(False)
        chassis.notifyCollisions(True)  # advance collision check, do callback in pg_collision_callback

        physics_world = get_engine().physics_world
        vehicle_chassis = BulletVehicle(physics_world.dynamic_world, chassis)
        vehicle_chassis.setCoordinateSystem(ZUp)
        self.dynamic_nodes.append(vehicle_chassis)
        return vehicle_chassis

    def _add_visualization(self):
        if self.render:
            [path, scale, x_y_z_offset, H] = self.path
            if path not in BaseVehicle.model_collection:
                car_model = self.loader.loadModel(AssetLoader.file_path("models", path, "vehicle.gltf"))
                BaseVehicle.model_collection[path] = car_model
            else:
                car_model = BaseVehicle.model_collection[path]
            car_model.setScale(scale)
            car_model.setH(H)
            car_model.setPos(x_y_z_offset)
            car_model.setZ(-self.TIRE_RADIUS - self.CHASSIS_TO_WHEEL_AXIS + x_y_z_offset[-1])
            car_model.instanceTo(self.origin)
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
        f_l = self.FRONT_WHEELBASE
        r_l = -self.REAR_WHEELBASE
        lateral = self.LATERAL_TIRE_TO_CENTER
        axis_height = self.TIRE_RADIUS - self.CHASSIS_TO_WHEEL_AXIS
        radius = self.TIRE_RADIUS
        wheels = []
        for k, pos in enumerate([Vec3(lateral, f_l, axis_height), Vec3(-lateral, f_l, axis_height),
                                 Vec3(lateral, r_l, axis_height), Vec3(-lateral, r_l, axis_height)]):
            wheel = self._add_wheel(pos, radius, True if k < 2 else False, True if k == 0 or k == 2 else False)
            wheels.append(wheel)
        return wheels

    def _add_wheel(self, pos: Vec3, radius: float, front: bool, left):
        wheel_np = self.origin.attachNewNode("wheel")
        if self.render:
            model = 'right_tire_front.gltf' if front else 'right_tire_back.gltf'
            model_path = AssetLoader.file_path("models", self.path[0], model)
            wheel_model = self.loader.loadModel(model_path)
            wheel_model.reparentTo(wheel_np)
            wheel_model.set_scale(1 if left else -1)
        wheel = self.system.create_wheel()
        wheel.setNode(wheel_np.node())
        wheel.setChassisConnectionPointCs(pos)
        wheel.setFrontWheel(front)
        wheel.setWheelDirectionCs(Vec3(0, 0, -1))
        wheel.setWheelAxleCs(Vec3(1, 0, 0))

        wheel.setWheelRadius(radius)
        wheel.setMaxSuspensionTravelCm(self.SUSPENSION_LENGTH)
        wheel.setSuspensionStiffness(self.SUSPENSION_STIFFNESS)
        wheel.setWheelsDampingRelaxation(4.8)
        wheel.setWheelsDampingCompression(1.2)
        wheel.setFrictionSlip(self.config["wheel_friction"])
        wheel.setRollInfluence(0.5)
        return wheel

    def add_image_sensor(self, name: str, sensor: ImageBuffer):
        self.image_sensors[name] = sensor

    def add_navigation(self):
        if not self.config["need_navigation"]:
            return
        navi = self.config["navigation_module"]
        if navi is None:
            navi = NodeNetworkNavigation if self.engine.current_map.road_network_type == NodeRoadNetwork \
                else EdgeNetworkNavigation
        self.navigation = \
            navi(self.engine,
                 show_navi_mark=self.engine.global_config["vehicle_config"]["show_navi_mark"],
                 random_navi_mark_color=self.engine.global_config["vehicle_config"]["random_navi_mark_color"],
                 show_dest_mark=self.engine.global_config["vehicle_config"]["show_dest_mark"],
                 show_line_to_dest=self.engine.global_config["vehicle_config"]["show_line_to_dest"],
                 panda_color=self.panda_color
                 )

    def update_map_info(self, map):
        """
        Update map info after reset()
        :param map: new map
        :return: None
        """
        if not self.config["need_navigation"]:
            return
        possible_lanes = ray_localization(
            self.heading, self.spawn_place, self.engine, return_all_result=True, use_heading_filter=False
        )
        possible_lane_indexes = [lane_index for lane, lane_index, dist in possible_lanes]
        try:
            idx = possible_lane_indexes.index(self.config["spawn_lane_index"])
        except ValueError:
            lane, new_l_index = possible_lanes[0][:-1]
        else:
            lane, new_l_index = possible_lanes[idx][:-1]
        dest = self.config["destination"]
        self.navigation.reset(
            map,
            current_lane=lane,
            destination=dest if dest is not None else None,
            random_seed=self.engine.global_random_seed
        )
        assert lane is not None, "spawn place is not on road!"
        self.navigation.update_localization(self)

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
            self.engine,
            self.position,
            np.rad2deg(self.heading_theta),
            self.LENGTH,
            self.WIDTH,
            CollisionGroup.Sidewalk,
            in_static_world=True if not self.render else False
        )
        if res.hasHit() and res.getNode().getName() == BodyName.Sidewalk:
            self.crash_sidewalk = True
            contacts.add(BodyName.Sidewalk)
        self.contact_results = contacts

    def destroy(self):
        super(BaseVehicle, self).destroy()
        if self.navigation is not None:
            self.navigation.destroy()
        self.navigation = None

        if self.side_detector is not None:
            self.side_detector.destroy()
            self.side_detector = None
        if self.lane_line_detector is not None:
            self.lane_line_detector.destroy()
            self.lane_line_detector = None
        if self.lidar is not None:
            self.lidar.destroy()
            self.lidar = None
        if len(self.image_sensors) != 0:
            for sensor in self.image_sensors.values():
                sensor.destroy()
        self.image_sensors = {}
        self.engine = None

    def set_heading_theta(self, heading_theta, rad_to_degree=True) -> None:
        """
        Set heading theta for this object
        :param heading_theta: float in rad
        """
        h = panda_heading(heading_theta)
        if rad_to_degree:
            h *= 180 / np.pi
        self.origin.setH(h - 90)

    def get_state(self):
        """
        Fetch more information
        """
        state = super(BaseVehicle, self).get_state()
        final_road = self.navigation.final_road
        state.update(
            {
                "steering": self.steering,
                "throttle_brake": self.throttle_brake,
                "crash_vehicle": self.crash_vehicle,
                "crash_object": self.crash_object,
                "crash_building": self.crash_building,
                "crash_sidewalk": self.crash_sidewalk
            }
        )
        if isinstance(self.navigation, NodeNetworkNavigation):
            state.update(
                {
                    "spawn_road": self.config["spawn_lane_index"][:-1],
                    "destination": (final_road.start_node, final_road.end_node)
                }
            )
        return state

    def set_state(self, state):
        super(BaseVehicle, self).set_state(state)
        self.throttle_brake = state["throttle_brake"]
        self.steering = state["steering"]

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
    def get_action_space_before_init(
        cls, extra_action_dim: int = 0, discrete_action=False, steering_dim=5, throttle_dim=5
    ):
        if not discrete_action:
            return gym.spaces.Box(-1.0, 1.0, shape=(2 + extra_action_dim, ), dtype=np.float32)
        else:
            return gym.spaces.MultiDiscrete([steering_dim, throttle_dim])

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

    def detach_from_world(self, physics_world):
        if self.navigation is not None:
            self.navigation.detach_from_world()
        if self.lidar is not None:
            self.lidar.detach_from_world()
        if self.side_detector is not None:
            self.side_detector.detach_from_world()
        if self.lane_line_detector is not None:
            self.lane_line_detector.detach_from_world()
        super(BaseVehicle, self).detach_from_world(physics_world)

    def attach_to_world(self, parent_node_path, physics_world):
        if self.config["show_navi_mark"] and self.config["need_navigation"]:
            self.navigation.attach_to_world(self.engine)
        if self.lidar is not None and self.config["show_lidar"]:
            self.lidar.attach_to_world(self.engine)
        if self.side_detector is not None and self.config["show_side_detector"]:
            self.side_detector.attach_to_world(self.engine)
        if self.lane_line_detector is not None and self.config["show_lane_line_detector"]:
            self.lane_line_detector.attach_to_world(self.engine)
        super(BaseVehicle, self).attach_to_world(parent_node_path, physics_world)

    def set_break_down(self, break_down=True):
        self.break_down = break_down
        # self.set_static(True)

    def convert_to_vehicle_coordinates(self, position, ego_heading=None, ego_position=None):
        """
        Give a world position, and convert it to vehicle coordinates
        The vehicle heading is X direction and right side is Y direction
        """
        # Projected to the heading of vehicle
        pos = ego_heading if ego_position is not None else self.position
        vector = position - pos
        forward = self.heading if ego_heading is None else ego_position

        norm_velocity = norm(forward[0], forward[1]) + 1e-6
        project_on_heading = (vector[0] * forward[0] + vector[1] * forward[1]) / norm_velocity

        side_direction = get_vertical_vector(forward)[1]
        side_norm = norm(side_direction[0], side_direction[1]) + 1e-6
        project_on_side = (vector[0] * side_direction[0] + vector[1] * side_direction[1]) / side_norm
        return project_on_heading, project_on_side

    def convert_to_world_coordinates(self, project_on_heading, project_on_side):
        """
        Give a position in vehicle coordinates, and convert it to world coordinates
        The vehicle heading is X direction and right side is Y direction
        """
        theta = np.arctan2(project_on_side, project_on_heading)
        theta = wrap_to_pi(self.heading_theta) + wrap_to_pi(theta)
        norm_len = norm(project_on_heading, project_on_side)
        position = self.position
        heading = np.sin(theta) * norm_len
        side = np.cos(theta) * norm_len
        return position[0] + side, position[1] + heading

    @property
    def max_speed(self):
        return self.config["max_speed"]

    @property
    def top_down_length(self):
        return self.LENGTH

    @property
    def top_down_width(self):
        return self.WIDTH

    @property
    def lane(self):
        return self.navigation.current_lane

    @property
    def lane_index(self):
        return self.navigation.current_lane.index

    @property
    def panda_color(self):
        c = super(BaseVehicle, self).panda_color
        if self._use_special_color:
            color = sns.color_palette("colorblind")
            rand_c = color[2]  # A pretty green
            c = rand_c
        return c
