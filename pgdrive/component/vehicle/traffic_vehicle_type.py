from pgdrive.component.vehicle.traffic_vehicle import TrafficVehicle

factor = 1


class LVehicle(TrafficVehicle):
    LENGTH = 4.8
    WIDTH = 1.8
    HEIGHT = 1.9
    path = [
        ['new/lada/scene.gltf', (factor * 1.1, factor * 1.1, factor * 1.1), (1.1, -13.5, factor * -0.046), 223],
    ]


class SVehicle(TrafficVehicle):
    LENGTH = 3.2
    WIDTH = 1.8
    HEIGHT = 1.5
    path = [
        ['new/beetle/scene.gltf', (factor * .008, factor * .006, factor * .0062), (-0.7, 0, factor * -0.16), -90],
    ]


class MVehicle(TrafficVehicle):
    LENGTH = 3.9
    WIDTH = 2.0
    HEIGHT = 1.3
    path = [
        ['new/130/scene.gltf', (factor * .0055, factor * .0046, factor * .0049), (0, 0, factor * 0.33), 90],
    ]


class XLVehicle(TrafficVehicle):
    LENGTH = 7.3
    WIDTH = 2.3
    HEIGHT = 2.7
    path = [['new/truck/scene.gltf', (factor * 0.031, factor * 0.025, factor * 0.025), (0.35, 0, factor * 0), 0]]


vehicle_type = {"s": SVehicle, "m": MVehicle, "l": LVehicle, "xl": XLVehicle}
