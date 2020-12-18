import copy
import logging
import math
from collections import deque
from os import path

import numpy as np
from panda3d.bullet import BulletVehicle, BulletBoxShape, BulletRigidBodyNode, ZUp, BulletWorld, BulletGhostNode
from panda3d.core import Vec3, TransformState, NodePath, LQuaternionf, BitMask32, PythonCallbackObject

from pgdrive.pg_config import PgConfig
from pgdrive.pg_config.body_name import BodyName
from pgdrive.pg_config.cam_mask import CamMask
from pgdrive.pg_config.parameter_space import Parameter, VehicleParameterSpace
from pgdrive.pg_config.pg_space import PgSpace
from pgdrive.scene_creator.blocks.block import Block
from pgdrive.scene_creator.ego_vehicle.vehicle_module.lidar import Lidar
from pgdrive.scene_creator.ego_vehicle.vehicle_module.routing_localization import RoutingLocalizationModule
from pgdrive.scene_creator.ego_vehicle.vehicle_module.vehicle_panel import VehiclePanel
from pgdrive.scene_creator.lanes.circular_lane import CircularLane
from pgdrive.scene_creator.lanes.lane import AbstractLane
from pgdrive.scene_creator.lanes.straight_lane import StraightLane
from pgdrive.scene_creator.map import Map
from pgdrive.scene_creator.pg_traffic_vehicle.traffic_vehicle import PgTrafficVehicle
from pgdrive.utils.asset_loader import AssetLoader
from pgdrive.utils.element import DynamicElement
from pgdrive.utils.math_utils import get_vertical_vector, norm, clip
from pgdrive.world.image_buffer import ImageBuffer
from pgdrive.world.pg_world import PgWorld
from pgdrive.world.terrain import Terrain


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
    PARAMETER_SPACE = PgSpace(VehicleParameterSpace.BASE_VEHICLE)  # it will not sample config from parameter space
    COLLISION_MASK = 1
    STEERING_INCREMENT = 0.05

    default_vehicle_config = PgConfig(
        dict(
            lidar=(240, 50, 4),  # laser num, distance, other vehicle info num
            mini_map=(84, 84, 250),  # buffer length, width
            rgb_cam=(84, 84),  # buffer length, width
            depth_cam=(84, 84, True),  # buffer length, width, view_ground
            show_navi_point=False,
            increment_steering=False,
            wheel_friction=0.6,
        )
    )

    born_place = (5, 0)
    LENGTH = None
    WIDTH = None

    def __init__(self, pg_world: PgWorld, vehicle_config: dict = None, random_seed: int = 0, config: dict = None):
        """
        This Vehicle Config is different from self.get_config(), and it is used to define which modules to use, and
        module parameters.
        """
        super(BaseVehicle, self).__init__(random_seed)
        self.pg_world = pg_world
        self.node_path = NodePath("vehicle")

        # config info
        self.set_config(self.PARAMETER_SPACE.sample())
        if config is not None:
            self.set_config(config)

        self.vehicle_config = self.get_vehicle_config(
            vehicle_config
        ) if vehicle_config is not None else self.default_vehicle_config
        self.increment_steering = self.vehicle_config["increment_steering"]
        self.max_speed = self.get_config()[Parameter.speed_max]
        self.max_steering = self.get_config()[Parameter.steering_max]

        # create
        self._add_chassis(pg_world.physics_world)
        self.wheels = self._create_wheel()

        # modules
        self.image_sensors = {}
        self.lidar = None
        self.routing_localization = None
        self.lane = None
        self.lane_index = None

        if (not self.pg_world.pg_config["highway_render"]) and (self.pg_world.mode != "none"):
            self.vehicle_panel = VehiclePanel(self, self.pg_world)
        else:
            self.vehicle_panel = None

        # other info
        self.throttle_brake = 0.0
        self.steering = 0
        self.last_current_action = deque([(0.0, 0.0), (0.0, 0.0)], maxlen=2)
        self.last_position = self.born_place
        self.last_heading_dir = self.heading

        # collision group
        self.pg_world.physics_world.setGroupCollisionFlag(self.COLLISION_MASK, Block.COLLISION_MASK, True)
        self.pg_world.physics_world.setGroupCollisionFlag(self.COLLISION_MASK, Terrain.COLLISION_MASK, True)
        self.pg_world.physics_world.setGroupCollisionFlag(self.COLLISION_MASK, self.COLLISION_MASK, False)
        self.pg_world.physics_world.setGroupCollisionFlag(self.COLLISION_MASK, PgTrafficVehicle.COLLISION_MASK, True)

        # done info
        self.crash = False
        self.out_of_road = False

        self.attach_to_pg_world(self.pg_world.pbr_render, self.pg_world.physics_world)

    @classmethod
    def get_vehicle_config(cls, new_config):
        default = copy.deepcopy(cls.default_vehicle_config)
        default.update(new_config)
        return default

    def prepare_step(self, action):
        """
        Save info before action
        """
        self.last_position = self.position
        self.last_heading_dir = self.heading
        self.last_current_action.append(action)  # the real step of physics world is implemented in taskMgr.step()
        if self.increment_steering:
            self.set_incremental_action(action)
        else:
            self.set_act(action)

    def update_state(self):
        if self.lidar is not None:
            self.lidar.perceive(self.position, self.heading_theta, self.pg_world.physics_world)
        if self.routing_localization is not None:
            self.lane, self.lane_index = self.routing_localization.update_navigation_localization(self)
        self._state_check()

    def reset(self, map: Map, pos: np.ndarray, heading: float):
        """
        pos is a 2-d array, and heading is a float (unit degree)
        """
        heading = np.deg2rad(heading - 90)
        self.chassis_np.setPos(Vec3(*pos, 1))
        self.chassis_np.setQuat(LQuaternionf(np.cos(heading / 2), 0, 0, np.sin(heading / 2)))
        self.update_map_info(map)
        self.chassis_np.node().clearForces()
        self.chassis_np.node().setLinearVelocity(Vec3(0, 0, 0))
        self.chassis_np.node().setAngularVelocity(Vec3(0, 0, 0))
        self.system.resetSuspension()

        # done info
        self.crash = False
        self.out_of_road = False

        # other info
        self.throttle_brake = 0.0
        self.steering = 0
        self.last_current_action = deque([(0.0, 0.0), (0.0, 0.0)], maxlen=2)
        self.last_position = self.born_place
        self.last_heading_dir = self.heading

        if "depth_cam" in self.image_sensors and self.image_sensors["depth_cam"].view_ground:
            for block in map.blocks:
                block.node_path.hide(CamMask.DepthCam)

    def get_state(self):
        pass

    def set_state(self, state):
        pass

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

    @property
    def position(self):
        pos_3d = self.chassis_np.getPos()
        return np.array([pos_3d[0], -pos_3d[1]])

    @property
    def speed(self):
        """
        km/h
        """
        speed = self.chassis_np.node().get_linear_velocity().length() * 3.6
        return clip(speed, 0.0, 100000.0)

    @property
    def heading(self):
        real_heading = self.heading_theta
        heading = np.array([np.cos(real_heading), np.sin(real_heading)])
        return heading

    @property
    def heading_theta(self):
        return -(self.chassis_np.getHpr()[0] + 90) / 180 * math.pi

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.velocity_direction

    @property
    def velocity_direction(self):
        direction = self.system.get_forward_vector()
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

    def _add_chassis(self, pg_physics_world: BulletWorld):
        para = self.get_config()
        chassis = BulletRigidBodyNode(BodyName.Ego_vehicle_top)
        chassis.setIntoCollideMask(BitMask32.bit(self.COLLISION_MASK))
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
        self.pg_world.physics_world.setContactAddedCallback(PythonCallbackObject(self._collision_check))
        self.bullet_nodes.append(chassis)

        chassis_beneath = BulletGhostNode(BodyName.Ego_vehicle)
        chassis_beneath.setIntoCollideMask(BitMask32.bit(self.COLLISION_MASK))
        chassis_beneath.addShape(chassis_shape)
        self.chassis_beneath_np = self.chassis_np.attachNewNode(chassis_beneath)
        self.bullet_nodes.append(chassis_beneath)

        self.system = BulletVehicle(pg_physics_world, chassis)
        self.system.setCoordinateSystem(ZUp)
        self.bullet_nodes.append(self.system)
        self.LENGTH = para[Parameter.vehicle_length]
        self.WIDTH = para[Parameter.vehicle_width]

        if self.render:
            model_path = 'models/ferra/scene.gltf'
            self.chassis_vis = self.loader.loadModel(path.join(AssetLoader.asset_path, model_path))
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
            model_path = 'models/yugo/yugotireR.egg' if left else 'models/yugo/yugotireL.egg'
            wheel_model = self.loader.loadModel(path.join(AssetLoader.asset_path, model_path))
            wheel_model.reparentTo(wheel_np)
            wheel_model.set_scale(1.4, radius / 0.25, radius / 0.25)
        wheel = self.system.create_wheel()
        wheel.setNode(wheel_np.node())
        wheel.setChassisConnectionPointCs(pos)
        wheel.setFrontWheel(front)
        wheel.setWheelDirectionCs(Vec3(0, 0, -1))
        wheel.setWheelAxleCs(Vec3(1, 0, 0))

        # TODO add them to PgConfig in the future
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

    def add_lidar(self, laser_num=240, distance=50):
        self.lidar = Lidar(self.pg_world.render, laser_num, distance)

    def add_routing_localization(self, show_navi_point: bool):
        self.routing_localization = RoutingLocalizationModule(show_navi_point)

    def update_map_info(self, map):
        self.routing_localization.update(map, self.pg_world.worldNP)
        self.lane_index = self.routing_localization.map.road_network.get_closest_lane_index((self.born_place))
        self.lane = self.routing_localization.map.road_network.get_lane(self.lane_index)

    def _state_check(self):
        """
        Check States and filter to update info
        """
        result = self.pg_world.physics_world.contactTest(self.chassis_beneath_np.node(), True)
        contacts = set()
        for contact in result.getContacts():
            node0 = contact.getNode0()
            node1 = contact.getNode1()
            name = [node0.getName(), node1.getName()]
            name.remove(BodyName.Ego_vehicle)
            if name[0] == "Ground":
                continue
            elif name[0] == BodyName.Side_walk:
                self.out_of_road = True
            contacts.add(name[0])
        if self.render:
            self.pg_world.render_collision_info(contacts)

    def _collision_check(self, contact):
        """
        It may lower the performance if overdone
        """
        node0 = contact.getNode0().getName()
        node1 = contact.getNode1().getName()
        name = [node0, node1]
        name.remove(BodyName.Ego_vehicle_top)
        if name[0] == BodyName.Traffic_vehicle:
            self.crash = True
            logging.debug("Crash with {}".format(name[0]))

    def destroy(self, pg_world: BulletWorld):
        self.pg_world.physics_world.clearContactAddedCallback()
        self.pg_world = None
        self.routing_localization = None
        if self.lidar is not None:
            self.lidar.destroy()
            self.lidar = None
        if len(self.image_sensors) != 0:
            for sensor in self.image_sensors.values():
                sensor.destroy(self.pg_world)
        self.image_sensors = None
        if self.vehicle_panel is not None:
            self.vehicle_panel.destroy()
        super(BaseVehicle, self).destroy(pg_world)

    def __del__(self):
        super(BaseVehicle, self).__del__()
        self.pg_world = None
        self.lidar = None
        self.mini_map = None
        self.rgb_cam = None
        self.routing_localization = None
        self.wheels = None
