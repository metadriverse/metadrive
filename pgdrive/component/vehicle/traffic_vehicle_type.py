from pgdrive.component.vehicle.traffic_vehicle import TrafficVehicle

factor = 1


class XLVehicle(TrafficVehicle):
    LENGTH = 5.8
    WIDTH = 2.3
    HEIGHT = 2.8
    # path = [['new/truck/scene.gltf', (factor * 0.031, factor * 0.025, factor * 0.025), (0.35, 0, factor * 0), 0]]
    path = [['new/truck/scene.gltf', (factor, factor, factor), (0, 0, 0), 0]]


class LVehicle(TrafficVehicle):
    LENGTH = 4.5
    WIDTH = 1.86
    HEIGHT = 1.85
    path = [
        # ['new/lada/scene.gltf', (factor * 1.1, factor * 1.1, factor * 1.1), (1.1, -13.5, factor * -0.046), 223],
        ['new/lada/scene.gltf', (factor, factor, factor), (0, 0, 0), 0],
    ]


class MVehicle(TrafficVehicle):
    LENGTH = 4.4
    WIDTH = 1.85
    HEIGHT = 1.37
    path = [
        # ['new/130/scene.gltf', (factor * .0055, factor * .0046, factor * .0049), (0, 0, factor * 0.33), 90],
        ['new/130/scene.gltf', (factor, factor, factor), (0, 0, 0), 0],
    ]


class SVehicle(TrafficVehicle):
    LENGTH = 4.25
    WIDTH = 1.7
    HEIGHT = 1.7
    path = [
        ['new/beetle/scene.gltf', (factor, factor, factor), (0, 0, 0), 0],
    ]


vehicle_type = {"s": SVehicle, "m": MVehicle, "l": LVehicle, "xl": XLVehicle}
