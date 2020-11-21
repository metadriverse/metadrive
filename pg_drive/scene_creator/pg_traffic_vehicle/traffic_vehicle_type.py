from pg_drive.scene_creator.pg_traffic_vehicle.traffic_vehicle import PgTrafficVehicle

factor = 1


class MVehicle(PgTrafficVehicle):
    LENGTH = 4.2
    WIDTH = 1.8
    HEIGHT = 1.7
    path = [
        ['new/lada/scene.gltf', (factor * 1.1, factor * 1.1, factor * 1.1), factor * -0.1, 223],
        ['new/beetle/scene.gltf', (factor * .007, factor * .007, factor * .006), factor * -0.15, -90],
    ]


class SVehicle(PgTrafficVehicle):
    LENGTH = 3.5
    WIDTH = 1.8
    HEIGHT = 1.5
    path = [
        ['new/130/scene.gltf', (factor * .0055, factor * .0055, factor * .0055), factor * 0.5, 90],
    ]


class LVehicle(PgTrafficVehicle):
    LENGTH = 8.0
    WIDTH = 2.2
    HEIGHT = 3.5
    path = [
        ['new/truck/scene.gltf', (factor * 0.025, factor * 0.025, factor * 0.025), factor * 0, 0] ]


car_type = {"s": SVehicle, "m": MVehicle, "l": LVehicle}
