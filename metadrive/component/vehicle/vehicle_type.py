from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.utils.space import ParameterSpace, VehicleParameterSpace

factor = 1


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
    path = ['vehicle/ferra/', (factor, factor, factor), (0, 0.0, 0.), 0]

    @property
    def LENGTH(self):
        return 4.51  # meters

    @property
    def HEIGHT(self):
        return 1.19  # meters

    @property
    def WIDTH(self):
        return 1.852  # meters


class StaticDefaultVehicle(DefaultVehicle):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.STATIC_DEFAULT_VEHICLE)


class XLVehicle(BaseVehicle):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.XL_VEHICLE)
    # LENGTH = 5.8
    # WIDTH = 2.3
    # HEIGHT = 2.8
    TIRE_RADIUS = 0.37
    REAR_WHEELBASE = 1.075
    FRONT_WHEELBASE = 1.726
    LATERAL_TIRE_TO_CENTER = 0.931
    CHASSIS_TO_WHEEL_AXIS = 0.3
    TIRE_WIDTH = 0.5
    MASS = 1600
    path = ['vehicle/truck/', (factor, factor, factor), (0, 0.3, 0.04), 0]

    @property
    def LENGTH(self):
        return 5.8  # meters

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
    TIRE_RADIUS = 0.39
    REAR_WHEELBASE = 1.10751
    FRONT_WHEELBASE = 1.391
    LATERAL_TIRE_TO_CENTER = 0.75
    TIRE_WIDTH = 0.35
    MASS = 1300
    path = ['vehicle/lada/', (factor, factor, factor), (0, -0.25, 0.07), 0]

    @property
    def LENGTH(self):
        return 4.5  # meters

    @property
    def HEIGHT(self):
        return 1.85  # meters

    @property
    def WIDTH(self):
        return 1.86  # meters


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

    path = ['vehicle/130/', (factor, factor, factor), (0, -0.05, 0.07), 0]

    @property
    def LENGTH(self):
        return 4.4  # meters

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
    FRONT_WHEELBASE = 1.4126
    REAR_WHEELBASE = 1.07
    TIRE_RADIUS = 0.376
    TIRE_WIDTH = 0.25
    MASS = 800
    path = ['vehicle/beetle/', (factor, factor, factor), (0, -0.2, 0.03), 0]

    @property
    def LENGTH(self):
        return 4.25  # meters

    @property
    def HEIGHT(self):
        return 1.70  # meters

    @property
    def WIDTH(self):
        return 1.70  # meters


vehicle_type = {
    "s": SVehicle,
    "m": MVehicle,
    "l": LVehicle,
    "xl": XLVehicle,
    "default": DefaultVehicle,
    "static_default": StaticDefaultVehicle
}


def random_vehicle_type(np_random, p=None):
    prob = [1 / len(vehicle_type) for _ in range(len(vehicle_type))] if p is None else p
    return vehicle_type[np_random.choice(list(vehicle_type.keys()), p=prob)]


class VaryingShapeVehicle(DefaultVehicle):
    @property
    def WIDTH(self):
        return self.config["width"] if self.config["width"] is not None else super(VaryingShapeVehicle, self).WIDTH

    @property
    def LENGTH(self):
        return self.config["length"] if self.config["length"] is not None else super(VaryingShapeVehicle, self).LENGTH

    @property
    def HEIGHT(self):
        return self.config["height"] if self.config["height"] is not None else super(VaryingShapeVehicle, self).HEIGHT

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
        if should_force_reset:
            self.destroy()
            self.__init__(
                vehicle_config=vehicle_config,
                name=self.name,
                random_seed=self.random_seed,
                position=position,
                heading=heading
            )

        return super(VaryingShapeVehicle, self).reset(
            random_seed=random_seed, vehicle_config=vehicle_config, position=position, heading=heading, *args, **kwargs
        )
