from metadrive.component.pg_space import ParameterSpace, VehicleParameterSpace
from metadrive.component.vehicle.base_vehicle import BaseVehicle


class DefaultVehicle(BaseVehicle):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.DEFAULT_VEHICLE)
    # LENGTH = 4.51
    # WIDTH = 1.852
    # HEIGHT = 1.19
    TIRE_RADIUS = 0.313
    TIRE_WIDTH = 0.25
    MASS = 1100
    LATERAL_TIRE_TO_CENTER = 0.815
    FRONT_WHEELBASE = 1.05234
    REAR_WHEELBASE = 1.4166
    path = ['vehicle/ferra/vehicle.gltf', (1, 1, 1), (0, 0.075, 0.), (0, 0, 0)]

    @property
    def LENGTH(self):
        return 4.515  # meters

    @property
    def HEIGHT(self):
        return 1.19  # meters

    @property
    def WIDTH(self):
        return 1.852  # meters


# When using DefaultVehicle as traffic, please use this class.


class TrafficDefaultVehicle(DefaultVehicle):
    pass


class StaticDefaultVehicle(DefaultVehicle):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.STATIC_DEFAULT_VEHICLE)


class XLVehicle(BaseVehicle):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.XL_VEHICLE)
    # LENGTH = 5.8
    # WIDTH = 2.3
    # HEIGHT = 2.8
    TIRE_RADIUS = 0.37
    TIRE_MODEL_CORRECT = -1
    REAR_WHEELBASE = 1.075
    FRONT_WHEELBASE = 1.726
    LATERAL_TIRE_TO_CENTER = 0.931
    CHASSIS_TO_WHEEL_AXIS = 0.3
    TIRE_WIDTH = 0.5
    MASS = 1600
    LIGHT_POSITION = (-0.75, 2.7, 0.2)
    path = ['vehicle/truck/vehicle.gltf', (1, 1, 1), (0, 0.25, 0.04), (0, 0, 0)]

    @property
    def LENGTH(self):
        return 5.74  # meters

    @property
    def HEIGHT(self):
        return 2.8  # meters

    @property
    def WIDTH(self):
        return 2.3  # meters


class LVehicle(BaseVehicle):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.L_VEHICLE)
    # LENGTH = 4.5
    # WIDTH = 1.86
    # HEIGHT = 1.85
    TIRE_RADIUS = 0.429
    REAR_WHEELBASE = 1.218261
    FRONT_WHEELBASE = 1.5301
    LATERAL_TIRE_TO_CENTER = 0.75
    TIRE_WIDTH = 0.35
    MASS = 1300
    LIGHT_POSITION = (-0.65, 2.13, 0.3)

    path = ['vehicle/lada/vehicle.gltf', (1.1, 1.1, 1.1), (0, -0.27, 0.07), (0, 0, 0)]

    @property
    def LENGTH(self):
        return 4.87  # meters

    @property
    def HEIGHT(self):
        return 1.85  # meters

    @property
    def WIDTH(self):
        return 2.046  # meters


class MVehicle(BaseVehicle):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.M_VEHICLE)
    # LENGTH = 4.4
    # WIDTH = 1.85
    # HEIGHT = 1.37
    TIRE_RADIUS = 0.39
    REAR_WHEELBASE = 1.203
    FRONT_WHEELBASE = 1.285
    LATERAL_TIRE_TO_CENTER = 0.803
    TIRE_WIDTH = 0.3
    MASS = 1200
    LIGHT_POSITION = (-0.67, 1.86, 0.22)

    path = ['vehicle/130/vehicle.gltf', (1, 1, 1), (0, -0.05, 0.1), (0, 0, 0)]

    @property
    def LENGTH(self):
        return 4.6  # meters

    @property
    def HEIGHT(self):
        return 1.37  # meters

    @property
    def WIDTH(self):
        return 1.85  # meters


class SVehicle(BaseVehicle):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.S_VEHICLE)
    # LENGTH = 4.25
    # WIDTH = 1.7
    # HEIGHT = 1.7
    LATERAL_TIRE_TO_CENTER = 0.7
    TIRE_TWO_SIDED = True
    FRONT_WHEELBASE = 1.385
    REAR_WHEELBASE = 1.11
    TIRE_RADIUS = 0.376
    TIRE_WIDTH = 0.25
    MASS = 800
    LIGHT_POSITION = (-0.57, 1.86, 0.23)

    @property
    def path(self):
        if self.use_render_pipeline:
            return [
                'vehicle/beetle/vehicle.bam', (0.0077, 0.0077, 0.0077), (0.04512, -0.24 - 0.04512, 1.77), (-90, -90, 0)
            ]
        else:
            factor = 1
            return ['vehicle/beetle/vehicle.gltf', (factor, factor, factor), (0, -0.2, 0.03), (0, 0, 0)]

    @property
    def LENGTH(self):
        return 4.3  # meters

    @property
    def HEIGHT(self):
        return 1.70  # meters

    @property
    def WIDTH(self):
        return 1.70  # meters


class VaryingDynamicsVehicle(DefaultVehicle):
    @property
    def WIDTH(self):
        return self.config["width"] if self.config["width"] is not None else super(VaryingDynamicsVehicle, self).WIDTH

    @property
    def LENGTH(self):
        return self.config["length"] if self.config["length"] is not None else super(
            VaryingDynamicsVehicle, self
        ).LENGTH

    @property
    def HEIGHT(self):
        return self.config["height"] if self.config["height"] is not None else super(
            VaryingDynamicsVehicle, self
        ).HEIGHT

    @property
    def MASS(self):
        return self.config["mass"] if self.config["mass"] is not None else super(VaryingDynamicsVehicle, self).MASS

    def reset(
        self,
        random_seed=None,
        vehicle_config=None,
        position=None,
        heading: float = 0.0,  # In degree!
        *args,
        **kwargs
    ):

        assert "width" not in self.PARAMETER_SPACE
        assert "height" not in self.PARAMETER_SPACE
        assert "length" not in self.PARAMETER_SPACE
        should_force_reset = False
        if vehicle_config is not None:
            if vehicle_config["width"] is not None and vehicle_config["width"] != self.WIDTH:
                should_force_reset = True
            if vehicle_config["height"] is not None and vehicle_config["height"] != self.HEIGHT:
                should_force_reset = True
            if vehicle_config["length"] is not None and vehicle_config["length"] != self.LENGTH:
                should_force_reset = True
            if "max_engine_force" in vehicle_config and \
                    vehicle_config["max_engine_force"] is not None and \
                    vehicle_config["max_engine_force"] != self.config["max_engine_force"]:
                should_force_reset = True
            if "max_brake_force" in vehicle_config and \
                    vehicle_config["max_brake_force"] is not None and \
                    vehicle_config["max_brake_force"] != self.config["max_brake_force"]:
                should_force_reset = True
            if "wheel_friction" in vehicle_config and \
                    vehicle_config["wheel_friction"] is not None and \
                    vehicle_config["wheel_friction"] != self.config["wheel_friction"]:
                should_force_reset = True
            if "max_steering" in vehicle_config and \
                    vehicle_config["max_steering"] is not None and \
                    vehicle_config["max_steering"] != self.config["max_steering"]:
                self.max_steering = vehicle_config["max_steering"]
                should_force_reset = True
            if "mass" in vehicle_config and \
                    vehicle_config["mass"] is not None and \
                    vehicle_config["mass"] != self.config["mass"]:
                should_force_reset = True

        # def process_memory():
        #     import psutil
        #     import os
        #     process = psutil.Process(os.getpid())
        #     mem_info = process.memory_info()
        #     return mem_info.rss
        #
        # cm = process_memory()

        if should_force_reset:
            self.destroy()
            self.__init__(
                vehicle_config=vehicle_config,
                name=self.name,
                random_seed=self.random_seed,
                position=position,
                heading=heading
            )

            # lm = process_memory()
            # print("{}:  Reset! Mem Change {:.3f}MB".format("1 Force Re-Init Vehicle", (lm - cm) / 1e6))
            # cm = lm

        assert self.max_steering == self.config["max_steering"]

        ret = super(VaryingDynamicsVehicle, self).reset(
            random_seed=random_seed, vehicle_config=vehicle_config, position=position, heading=heading, *args, **kwargs
        )

        # lm = process_memory()
        # print("{}:  Reset! Mem Change {:.3f}MB".format("2 Force Reset Vehicle", (lm - cm) / 1e6))
        # cm = lm

        return ret


def random_vehicle_type(np_random, p=None):
    prob = [1 / len(vehicle_type) for _ in range(len(vehicle_type))] if p is None else p
    return vehicle_type[np_random.choice(list(vehicle_type.keys()), p=prob)]


vehicle_type = {
    "s": SVehicle,
    "m": MVehicle,
    "l": LVehicle,
    "xl": XLVehicle,
    "default": DefaultVehicle,
    "static_default": StaticDefaultVehicle,
    "varying_dynamics": VaryingDynamicsVehicle
}

VaryingShapeVehicle = VaryingDynamicsVehicle

type_count = [0 for i in range(3)]


def reset_vehicle_type_count(np_random=None):
    global type_count
    if np_random is None:
        type_count = [0 for i in range(3)]
    else:
        type_count = [np_random.randint(100) for i in range(3)]


def get_vehicle_type(length, np_random=None, need_default_vehicle=False):
    if np_random is not None:
        if length <= 4:
            return SVehicle
        elif length <= 5.5:
            return [LVehicle, SVehicle, MVehicle][np_random.randint(3)]
        else:
            return [LVehicle, XLVehicle][np_random.randint(2)]
    else:
        global type_count
        # evenly sample
        if length <= 4:
            return SVehicle
        elif length <= 5.5:
            type_count[1] += 1
            vs = [MVehicle, LVehicle, SVehicle]
            if need_default_vehicle:
                vs.append(TrafficDefaultVehicle)
            return vs[type_count[1] % len(vs)]
        else:
            type_count[2] += 1
            vs = [LVehicle, XLVehicle]
            return vs[type_count[2] % len(vs)]
