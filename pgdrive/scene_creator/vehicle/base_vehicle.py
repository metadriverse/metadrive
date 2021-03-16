import copy
import logging
import math
import time
from collections import deque
from typing import Optional
from pgdrive.rl_utils.cost import pg_cost_scheme
from pgdrive.rl_utils.reward import pg_reward_scheme
import gym
import numpy as np
from panda3d.bullet import BulletVehicle, BulletBoxShape, BulletRigidBodyNode, ZUp, BulletGhostNode
from panda3d.core import Vec3, TransformState, NodePath, LQuaternionf, BitMask32, PythonCallbackObject, TextNode
from pgdrive.constants import RENDER_MODE_ONSCREEN, COLOR, COLLISION_INFO_COLOR, BodyName
from pgdrive.pg_config import PGConfig
from pgdrive.pg_config.cam_mask import CamMask
from pgdrive.pg_config.collision_group import CollisionGroup
from pgdrive.pg_config.parameter_space import Parameter, VehicleParameterSpace
from pgdrive.pg_config.pg_space import PGSpace
from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.scene_creator.lane.abs_lane import AbstractLane
from pgdrive.scene_creator.lane.circular_lane import CircularLane
from pgdrive.scene_creator.lane.straight_lane import StraightLane
from pgdrive.scene_creator.map import Map
from pgdrive.scene_creator.vehicle_module import Lidar, MiniMap
from pgdrive.scene_creator.vehicle_module.depth_camera import DepthCamera
from pgdrive.scene_creator.vehicle_module.rgb_camera import RGBCamera
from pgdrive.scene_creator.vehicle_module.routing_localization import RoutingLocalizationModule
from pgdrive.scene_creator.vehicle_module.vehicle_panel import VehiclePanel
from pgdrive.utils import safe_clip
from pgdrive.utils.asset_loader import AssetLoader
from pgdrive.utils.coordinates_shift import panda_position, pgdrive_position, panda_heading, pgdrive_heading
from pgdrive.utils.element import DynamicElement
from pgdrive.utils.math_utils import get_vertical_vector, norm, clip
from pgdrive.utils.scene_utils import ray_localization
from pgdrive.world.image_buffer import ImageBuffer
from pgdrive.world.pg_physics_world import PGPhysicsWorld
from pgdrive.world.pg_world import PGWorld


class BaseVehicle(DynamicElement):
    Ego_state_obs_dim = 9
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
    PARAMETER_SPACE = PGSpace(VehicleParameterSpace.BASE_VEHICLE)  # it will not sample config from parameter space
    COLLISION_MASK = CollisionGroup.EgoVehicle
    STEERING_INCREMENT = 0.05

    @classmethod
    def _default_vehicle_config(cls) -> PGConfig:
        vehicle_config = dict(
            # ===== vehicle module config =====
            lidar=dict(num_lasers=240, distance=50, num_others=4),  # laser num, distance, other vehicle info num
            show_lidar=False,
            mini_map=(84, 84, 250),  # buffer length, width
            rgb_cam=(84, 84),  # buffer length, width
            depth_cam=(84, 84, True),  # buffer length, width, view_ground
            show_navi_mark=True,
            increment_steering=False,
            wheel_friction=0.6,

            # ===== use image =====
            image_source="rgb_cam",  # take effect when only when use_image == True
            use_image=False,
            rgb_clip=True,

            # ===== vehicle born =====
            born_lane_index=(FirstBlock.NODE_1, FirstBlock.NODE_2, 0),
            born_longitude=5.0,
            born_lateral=0.0,

            # ==== others ====
            overtake_stat=False,  # we usually set to True when evaluation
            action_check=False,
            use_saver=False,
            save_level=0.5,
        )
        vehicle_config.update(pg_reward_scheme)
        vehicle_config.update(pg_cost_scheme)
        return PGConfig(vehicle_config)

    LENGTH = None
    WIDTH = None

    def __init__(
        self, pg_world: PGWorld, vehicle_config: dict = None, physics_config: dict = None, random_seed: int = 0
    ):
        """
        This Vehicle Config is different from self.get_config(), and it is used to define which modules to use, and
        module parameters. And self.physics_config defines the physics feature of vehicles, such as length/width
        :param pg_world: PGWorld
        :param vehicle_config: mostly, vehicle module config
        :param physics_config: vehicle height/width/length, find more physics para in VehicleParameterSpace
        :param random_seed: int
        """

        self.vehicle_config = self.get_vehicle_config(vehicle_config) \
            if vehicle_config is not None else self._default_vehicle_config()

        # observation, action
        self.action_space = self.get_action_space_before_init()

        super(BaseVehicle, self).__init__(random_seed)
        # config info
        self.set_config(self.PARAMETER_SPACE.sample())
        if physics_config is not None:
            self.set_config(physics_config)
        self.increment_steering = self.vehicle_config["increment_steering"]
        self.max_speed = self.get_config()[Parameter.speed_max]
        self.max_steering = self.get_config()[Parameter.steering_max]

        self.pg_world = pg_world
        self.node_path = NodePath("vehicle")

        # create
        self.born_place = (0, 0)
        self._add_chassis(pg_world.physics_world)
        self.wheels = self._create_wheel()

        # modules
        self.image_sensors = {}
        self.lidar: Optional[Lidar] = None
        self.routing_localization: Optional[RoutingLocalizationModule] = None
        self.lane: Optional[AbstractLane] = None
        self.lane_index = None
        self.vehicle_panel = VehiclePanel(self.pg_world) if (self.pg_world.mode == RENDER_MODE_ONSCREEN) else None

        # state info
        self.throttle_brake = 0.0
        self.steering = 0
        self.last_current_action = deque([(0.0, 0.0), (0.0, 0.0)], maxlen=2)
        self.last_position = self.born_place
        self.last_heading_dir = self.heading
        self.dist_to_left = None
        self.dist_to_right = None

        # collision info render
        self.collision_info_np = self._init_collision_info_render(pg_world)
        self.collision_banners = {}  # to save time
        self.current_banner = None
        self.attach_to_pg_world(self.pg_world.pbr_render, self.pg_world.physics_world)

        # step info
        self.crash_vehicle = None
        self.crash_object = None
        self.out_of_route = None
        self.crash_side_walk = None
        self.on_lane = None
        self.step_info = None
        self._init_step_info()

        # others
        self._frame_objects_crashed = []  # inner loop, object will only be crashed for once
        self._add_modules_for_vehicle(pg_world.pg_config["use_render"])
        self.takeover = False
        self._expert_takeover = False

        # overtake_stat
        self.front_vehicles = set()
        self.back_vehicles = set()

    def _add_modules_for_vehicle(self, use_render: bool):
        # add self module for training according to config
        vehicle_config = self.vehicle_config
        self.add_routing_localization(vehicle_config["show_navi_mark"])  # default added
        if not self.vehicle_config["use_image"]:
            if vehicle_config["lidar"]["num_lasers"] > 0 and vehicle_config["lidar"]["distance"] > 0:
                self.add_lidar(
                    num_lasers=vehicle_config["lidar"]["num_lasers"],
                    distance=vehicle_config["lidar"]["distance"],
                    show_lidar_point=vehicle_config["show_lidar"]
                )
            else:
                import logging
                logging.warning(
                    "You have set the lidar config to: {}, which seems to be invalid!".format(vehicle_config["lidar"])
                )

            if use_render:
                rgb_cam_config = vehicle_config["rgb_cam"]
                rgb_cam = RGBCamera(rgb_cam_config[0], rgb_cam_config[1], self.chassis_np, self.pg_world)
                self.add_image_sensor("rgb_cam", rgb_cam)

                mini_map = MiniMap(vehicle_config["mini_map"], self.chassis_np, self.pg_world)
                self.add_image_sensor("mini_map", mini_map)
            return

        if vehicle_config["use_image"]:
            # 3 types image observation
            if vehicle_config["image_source"] == "rgb_cam":
                rgb_cam_config = vehicle_config["rgb_cam"]
                rgb_cam = RGBCamera(rgb_cam_config[0], rgb_cam_config[1], self.chassis_np, self.pg_world)
                self.add_image_sensor("rgb_cam", rgb_cam)
            elif vehicle_config["image_source"] == "mini_map":
                mini_map = MiniMap(vehicle_config["mini_map"], self.chassis_np, self.pg_world)
                self.add_image_sensor("mini_map", mini_map)
            elif vehicle_config["image_source"] == "depth_cam":
                cam_config = vehicle_config["depth_cam"]
                depth_cam = DepthCamera(*cam_config, self.chassis_np, self.pg_world)
                self.add_image_sensor("depth_cam", depth_cam)
            else:
                raise ValueError("No module named {}".format(vehicle_config["image_source"]))

        # load more sensors for visualization when render, only for beauty...
        if use_render:
            if vehicle_config["image_source"] == "mini_map":
                rgb_cam_config = vehicle_config["rgb_cam"]
                rgb_cam = RGBCamera(rgb_cam_config[0], rgb_cam_config[1], self.chassis_np, self.pg_world)
                self.add_image_sensor("rgb_cam", rgb_cam)
            else:
                mini_map = MiniMap(vehicle_config["mini_map"], self.chassis_np, self.pg_world)
                self.add_image_sensor("mini_map", mini_map)

    def _init_step_info(self):
        # done info will be initialized every frame
        self.crash_vehicle = False
        self.crash_object = False
        self.crash_side_walk = False
        self.out_of_route = False  # re-route is required if is false
        self.on_lane = True  # on lane surface or not
        self.step_info = {"reward": 0, "cost": 0}

    @classmethod
    def get_vehicle_config(cls, new_config=None):
        default = copy.deepcopy(cls._default_vehicle_config())
        if new_config is None:
            return default
        default.update(new_config)
        return default

    def _preprocess_action(self, action):
        self.step_info["raw_action"] = (action[0], action[1])
        if self.vehicle_config["action_check"]:
            assert self.action_space.contains(action), "Input {} is not compatible with action space {}!".format(
                action, self.action_space
            )

        # protect agent from nan error
        action = safe_clip(action, min_val=self.action_space.low[0], max_val=self.action_space.high[0])
        return action

    def prepare_step(self, action):
        """
        Save info and make decision before action
        """
        # init step info to store info before each step
        self._init_step_info()
        action = self._preprocess_action(action)

        self._frame_objects_crashed = []
        self.last_position = self.position
        self.last_heading_dir = self.heading
        self.last_current_action.append(action)  # the real step of physics world is implemented in taskMgr.step()
        if self.increment_steering:
            self.set_incremental_action(action)
        else:
            self.set_act(action)
        if self.vehicle_panel is not None:
            self.vehicle_panel.renew_2d_car_para_visualization(self)

    def update_state(self, pg_world=None):
        # callback
        for obj in self._frame_objects_crashed:
            if obj.COST_ONCE:
                obj.crashed = True
        # lidar
        if self.lidar is not None:
            self.lidar.perceive(self.position, self.heading_theta, self.pg_world.physics_world)
        if self.routing_localization is not None:
            self.lane, self.lane_index = self.routing_localization.update_navigation_localization(self)
        self._state_check()
        self.update_dist_to_left_right()
        self.out_of_route = True if self.dist_to_right < 0 or self.dist_to_left < 0 else False
        self._update_overtake_stat()
        self.step_info.update(
            {
                "velocity": float(self.speed),
                "steering": float(self.steering),
                "acceleration": float(self.throttle_brake)
            }
        )

    def reset(self, map: Map, pos: np.ndarray = None, heading: float = 0.0):
        """
        pos is a 2-d array, and heading is a float (unit degree)
        if pos is not None, vehicle will be reset to the position
        else, vehicle will be reset to born place
        """
        if pos is None:
            self.born_place = map.road_network.get_lane(
                self.vehicle_config["born_lane_index"]
            ).position(self.vehicle_config["born_longitude"], self.vehicle_config["born_lateral"])
            pos = self.born_place
        heading = -np.deg2rad(heading) - np.pi / 2
        self.chassis_np.setPos(Vec3(*pos, 1))
        self.chassis_np.setQuat(LQuaternionf(np.cos(heading / 2), 0, 0, np.sin(heading / 2)))
        self.update_map_info(map)
        self.chassis_np.node().clearForces()
        self.chassis_np.node().setLinearVelocity(Vec3(0, 0, 0))
        self.chassis_np.node().setAngularVelocity(Vec3(0, 0, 0))
        self.system.resetSuspension()

        # done info
        self._init_step_info()

        # other info
        self.throttle_brake = 0.0
        self.steering = 0
        self.last_current_action = deque([(0.0, 0.0), (0.0, 0.0)], maxlen=2)
        self.last_position = self.born_place
        self.last_heading_dir = self.heading
        self.update_dist_to_left_right()
        self.takeover = False

        # overtake_stat
        self.front_vehicles = set()
        self.back_vehicles = set()

        # for render
        if self.vehicle_panel is not None:
            self.vehicle_panel.renew_2d_car_para_visualization(self)

        if "depth_cam" in self.image_sensors and self.image_sensors["depth_cam"].view_ground:
            for block in map.blocks:
                block.node_path.hide(CamMask.DepthCam)

    """------------------------------------------- act -------------------------------------------------"""

    def set_act(self, action):
        para = self.get_config()
        steering = action[0]
        self.throttle_brake = action[1]
        self.steering = steering
        self.system.setSteeringValue(self.steering * para[Parameter.steering_max], 0)
        self.system.setSteeringValue(self.steering * para[Parameter.steering_max], 1)
        self._apply_throttle_brake(action[1])

    def set_incremental_action(self, action: np.ndarray):
        self.throttle_brake = action[1]
        self.steering += action[0] * self.STEERING_INCREMENT
        self.steering = clip(self.steering, -1, 1)
        steering = self.steering * self.max_steering
        self.system.setSteeringValue(steering, 0)
        self.system.setSteeringValue(steering, 1)
        self._apply_throttle_brake(action[1])

    def _apply_throttle_brake(self, throttle_brake):
        para = self.get_config()
        max_engine_force = para[Parameter.engine_force_max]
        max_brake_force = para[Parameter.brake_force_max]
        for wheel_index in range(4):
            if throttle_brake >= 0:
                self.system.setBrake(2.0, wheel_index)
                if self.speed > self.max_speed:
                    self.system.applyEngineForce(0.0, wheel_index)
                else:
                    self.system.applyEngineForce(max_engine_force * throttle_brake, wheel_index)
            else:
                self.system.applyEngineForce(0.0, wheel_index)
                self.system.setBrake(abs(throttle_brake) * max_brake_force, wheel_index)

    """---------------------------------------- vehicle info ----------------------------------------------"""

    def update_dist_to_left_right(self):
        current_reference_lane = self.routing_localization.current_ref_lanes[-1]
        _, lateral_to_reference = current_reference_lane.local_coordinates(self.position)

        lateral_to_right = abs(
            lateral_to_reference) + self.routing_localization.map.lane_width / 2 if lateral_to_reference < 0 \
            else self.routing_localization.map.lane_width / 2 - abs(lateral_to_reference)

        lateral_to_left = self.routing_localization.map.lane_width * self.routing_localization.map.lane_num - lateral_to_right
        self.dist_to_left, self.dist_to_right = lateral_to_left, lateral_to_right

    @property
    def position(self):
        return pgdrive_position(self.chassis_np.getPos())

    @property
    def speed(self):
        """
        km/h
        """
        velocity = self.chassis_np.node().get_linear_velocity()
        speed = norm(velocity[0], velocity[1]) * 3.6
        return clip(speed, 0.0, 100000.0)

    @property
    def heading(self):
        real_heading = self.heading_theta
        heading = np.array([np.cos(real_heading), np.sin(real_heading)])
        return heading

    @property
    def heading_theta(self):
        """
        Get the heading theta of vehicle, unit [rad]
        :return:  heading in rad
        """
        return (pgdrive_heading(self.chassis_np.getH()) - 90) / 180 * math.pi

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.velocity_direction

    @property
    def velocity_direction(self):
        direction = self.system.getForwardVector()
        return np.asarray([direction[0], -direction[1]])

    @property
    def forward_direction(self):
        raise ValueError("This function id deprecated.")
        # print("This function id deprecated.")
        # direction = self.vehicle.get_forward_vector()
        # return np.array([direction[0], -direction[1]])

    @property
    def current_road(self):
        return self.lane_index[0:-1]

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
        else:
            raise ValueError("Unknown target lane type: {}".format(type(target_lane)))
        lateral_norm = norm(lateral[0], lateral[1])
        forward_direction = self.heading
        # print(f"Old forward direction: {self.forward_direction}, new heading {self.heading}")
        forward_direction_norm = norm(forward_direction[0], forward_direction[1])
        if not lateral_norm * forward_direction_norm:
            return 0
        # cos = self.forward_direction.dot(lateral) / (np.linalg.norm(lateral) * np.linalg.norm(self.forward_direction))
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
        assert self.routing_localization is not None, "a routing and localization module should be added " \
                                                      "to interact with other vehicles"
        if not vehicle:
            return np.nan
        if not lane:
            lane = self.lane
        return lane.local_coordinates(vehicle.position)[0] - lane.local_coordinates(self.position)[0]

    """-------------------------------------- for vehicle making ------------------------------------------"""

    def _add_chassis(self, pg_physics_world: PGPhysicsWorld):
        para = self.get_config()
        chassis = BulletRigidBodyNode(BodyName.Ego_vehicle)
        chassis.setIntoCollideMask(BitMask32.bit(CollisionGroup.EgoVehicle))
        chassis_shape = BulletBoxShape(
            Vec3(
                para[Parameter.vehicle_width] / 2, para[Parameter.vehicle_length] / 2,
                para[Parameter.vehicle_height] / 2
            )
        )
        ts = TransformState.makePos(Vec3(0, 0, para[Parameter.chassis_height] * 2))
        chassis.addShape(chassis_shape, ts)
        heading = np.deg2rad(-para[Parameter.heading] - 90)
        chassis.setMass(para[Parameter.mass])
        self.chassis_np = self.node_path.attachNewNode(chassis)
        # not random born now
        self.chassis_np.setPos(Vec3(*self.born_place, 1))
        self.chassis_np.setQuat(LQuaternionf(np.cos(heading / 2), 0, 0, np.sin(heading / 2)))
        chassis.setDeactivationEnabled(False)
        chassis.notifyCollisions(True)  # advance collision check
        self.pg_world.physics_world.dynamic_world.setContactAddedCallback(PythonCallbackObject(self._collision_check))
        self.dynamic_nodes.append(chassis)

        chassis_beneath = BulletGhostNode(BodyName.Ego_vehicle_beneath)
        chassis_beneath.setIntoCollideMask(BitMask32.bit(CollisionGroup.EgoVehicleBeneath))
        chassis_beneath.addShape(chassis_shape)
        self.chassis_beneath_np = self.chassis_np.attachNewNode(chassis_beneath)
        self.dynamic_nodes.append(chassis_beneath)

        self.system = BulletVehicle(pg_physics_world.dynamic_world, chassis)
        self.system.setCoordinateSystem(ZUp)
        self.dynamic_nodes.append(self.system)  # detach chassis will also detach system, so a waring will generate
        self.LENGTH = para[Parameter.vehicle_length]
        self.WIDTH = para[Parameter.vehicle_width]

        if self.render:
            model_path = 'models/ferra/scene.gltf'
            self.chassis_vis = self.loader.loadModel(AssetLoader.file_path(model_path))
            self.chassis_vis.setZ(para[Parameter.vehicle_vis_z])
            self.chassis_vis.setY(para[Parameter.vehicle_vis_y])
            self.chassis_vis.setH(para[Parameter.vehicle_vis_h])
            self.chassis_vis.set_scale(para[Parameter.vehicle_vis_scale])
            self.chassis_vis.reparentTo(self.chassis_np)

    def _create_wheel(self):
        para = self.get_config()
        f_l = para[Parameter.front_tire_longitude]
        r_l = -para[Parameter.rear_tire_longitude]
        lateral = para[Parameter.tire_lateral]
        axis_height = para[Parameter.tire_radius] + 0.05
        radius = para[Parameter.tire_radius]
        wheels = []
        for k, pos in enumerate([Vec3(lateral, f_l, axis_height), Vec3(-lateral, f_l, axis_height),
                                 Vec3(lateral, r_l, axis_height), Vec3(-lateral, r_l, axis_height)]):
            wheel = self._add_wheel(pos, radius, True if k < 2 else False, True if k == 0 or k == 2 else False)
            wheels.append(wheel)
        return wheels

    def _add_wheel(self, pos: Vec3, radius: float, front: bool, left):
        wheel_np = self.node_path.attachNewNode("wheel")
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

        # TODO add them to PGConfig in the future
        wheel.setWheelRadius(radius)
        wheel.setMaxSuspensionTravelCm(40)
        wheel.setSuspensionStiffness(30)
        wheel.setWheelsDampingRelaxation(4.8)
        wheel.setWheelsDampingCompression(1.2)
        wheel.setFrictionSlip(self.vehicle_config["wheel_friction"])
        wheel.setRollInfluence(1.5)
        return wheel

    def add_image_sensor(self, name: str, sensor: ImageBuffer):
        self.image_sensors[name] = sensor

    def add_lidar(self, num_lasers=240, distance=50, show_lidar_point=False):
        assert num_lasers > 0
        assert distance > 0
        self.lidar = Lidar(self.pg_world.render, num_lasers, distance, show_lidar_point)

    def add_routing_localization(self, show_navi_point: bool):
        self.routing_localization = RoutingLocalizationModule(self.pg_world, show_navi_point)

    def update_map_info(self, map):
        """
        Update map info after reset()
        :param map: new map
        :return: None
        """
        self.routing_localization.update(map)
        lane, new_l_index = ray_localization(np.array(self.born_place), self.pg_world)
        assert lane is not None, "Born place is not on road!"
        self.lane_index = new_l_index
        self.lane = lane

    def _state_check(self):
        """
        Check States and filter to update info
        """
        result = self.pg_world.physics_world.dynamic_world.contactTest(self.chassis_beneath_np.node(), True)
        contacts = set()
        for contact in result.getContacts():
            node0 = contact.getNode0()
            node1 = contact.getNode1()
            name = [node0.getName(), node1.getName()]
            name.remove(BodyName.Ego_vehicle_beneath)
            if name[0] == "Ground" or name[0] == BodyName.Lane:
                continue
            elif name[0] == BodyName.Side_walk:
                self.crash_side_walk = True
            contacts.add(name[0])
        if self.render:
            self.render_collision_info(contacts)

    def _collision_check(self, contact):
        """
        It may lower the performance if overdone
        """
        node0 = contact.getNode0().getName()
        node1 = contact.getNode1().getName()
        name = [node0, node1]
        name.remove(BodyName.Ego_vehicle)
        if name[0] in [BodyName.Traffic_vehicle, BodyName.Ego_vehicle]:
            self.crash_vehicle = True
        elif name[0] in [BodyName.Traffic_cone, BodyName.Traffic_triangle]:
            node = contact.getNode0() if contact.getNode0().hasPythonTag(name[0]) else contact.getNode1()
            self.crash_object = True if not node.getPythonTag(name[0]).crashed else False
            self._frame_objects_crashed.append(node.getPythonTag(name[0]))
        logging.debug("Crash with {}".format(name[0]))

    @staticmethod
    def _init_collision_info_render(pg_world):
        if pg_world.mode == "onscreen":
            info_np = NodePath("Collision info nodepath")
            info_np.reparentTo(pg_world.aspect2d)
        else:
            info_np = None
        return info_np

    def render_collision_info(self, contacts):
        contacts = sorted(list(contacts), key=lambda c: COLLISION_INFO_COLOR[COLOR[c]][0])
        text = contacts[0] if len(contacts) != 0 else None
        if text is None:
            text = "Normal" if time.time() - self.pg_world._episode_start_time > 10 else "Press H to see help message"
            self.render_banner(text, COLLISION_INFO_COLOR["green"][1])
        else:
            self.render_banner(text, COLLISION_INFO_COLOR[COLOR[text]][1])

    def render_banner(self, text, color=COLLISION_INFO_COLOR["green"][1]):
        """
        Render the banner in the left bottom corner.
        """
        if self.collision_info_np is None:
            return
        if self.current_banner is not None:
            self.current_banner.detachNode()
        if text in self.collision_banners:
            self.collision_banners[text].reparentTo(self.collision_info_np)
            self.current_banner = self.collision_banners[text]
        else:
            new_banner = NodePath(TextNode("collision_info:{}".format(text)))
            self.collision_banners[text] = new_banner
            text_node = new_banner.node()
            text_node.setCardColor(color)
            text_node.setText(text)
            text_node.setCardActual(-5 * self.pg_world.w_scale, 5.1 * self.pg_world.w_scale, -0.3, 1)
            text_node.setCardDecal(True)
            text_node.setTextColor(1, 1, 1, 1)
            text_node.setAlign(TextNode.A_center)
            new_banner.setScale(0.05)
            new_banner.setPos(-0.75 * self.pg_world.w_scale, 0, -0.8 * self.pg_world.h_scale)
            new_banner.reparentTo(self.collision_info_np)
            self.current_banner = new_banner

    def destroy(self, _=None):
        self.dynamic_nodes.remove(self.chassis_np.node())
        super(BaseVehicle, self).destroy(self.pg_world)
        self.pg_world.physics_world.dynamic_world.clearContactAddedCallback()
        self.routing_localization.destroy()
        self.routing_localization = None
        if self.lidar is not None:
            self.lidar.destroy()
            self.lidar = None
        if len(self.image_sensors) != 0:
            for sensor in self.image_sensors.values():
                sensor.destroy(self.pg_world)
        self.image_sensors = None
        if self.vehicle_panel is not None:
            self.vehicle_panel.destroy(self.pg_world)
        self.pg_world = None

    def set_position(self, position):
        """
        Should only be called when restore traffic from episode data
        :param position: 2d array or list
        :return: None
        """
        self.chassis_np.setPos(panda_position(position, 0.4))

    def set_heading(self, heading_theta) -> None:
        """
        Should only be called when restore traffic from episode data
        :param heading_theta: float in rad
        :return: None
        """
        self.chassis_np.setH((panda_heading(heading_theta) * 180 / np.pi) - 90)

    def get_state(self):
        return {
            "heading": self.heading_theta,
            "position": self.position.tolist(),
            "done": self.crash_vehicle or self.out_of_route or self.crash_side_walk or not self.on_lane
        }

    def set_state(self, state: dict):
        self.set_heading(state["heading"])
        self.set_position(state["position"])

    def _update_overtake_stat(self):
        if self.vehicle_config["overtake_stat"]:
            surrounding_vs = self.lidar.get_surrounding_vehicles()
            routing = self.routing_localization
            ckpt_idx = routing.target_checkpoints_index
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
        self.step_info["overtake_vehicle_num"] = self.get_overtake_num()

    def get_overtake_num(self):
        return len(self.front_vehicles.intersection(self.back_vehicles))

    @classmethod
    def get_action_space_before_init(cls):
        return gym.spaces.Box(-1.0, 1.0, shape=(2, ), dtype=np.float32)

    def remove_display_region(self):
        if self.vehicle_panel is not None:
            self.vehicle_panel.remove_display_region(self.pg_world)
        for sensor in self.image_sensors.values():
            sensor.remove_display_region(self.pg_world)

    def add_to_display(self):
        if self.vehicle_panel is not None:
            self.vehicle_panel.add_to_display(self.pg_world, self.vehicle_panel.default_region)
        for sensor in self.image_sensors.values():
            sensor.add_to_display(self.pg_world, sensor.default_region)

    def __del__(self):
        super(BaseVehicle, self).__del__()
        self.pg_world = None
        self.lidar = None
        self.mini_map = None
        self.rgb_cam = None
        self.routing_localization = None
        self.wheels = None
