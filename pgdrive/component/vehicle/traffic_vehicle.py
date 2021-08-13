from pgdrive.component.vehicle.base_vehicle import BaseVehicle
from pgdrive.utils.space import Parameter
from pgdrive.constants import CollisionGroup
from pgdrive.engine.asset_loader import AssetLoader


class TrafficVehicle(BaseVehicle):
    COLLISION_MASK = CollisionGroup.Vehicle
    HEIGHT = 1.8
    LENGTH = 4
    WIDTH = 2
    path = None
    break_down = False
    model_collection = {}  # save memory, load model once

    def __init__(self, vehicle_config, random_seed=None):
        """
        A traffic vehicle class.
        """
        super(TrafficVehicle, self).__init__(vehicle_config, random_seed=random_seed)

    def _add_visualization(self):
        [path, scale, x_y_z_offset, H] = self.path[self.np_random.randint(0, len(self.path))]
        if self.render:
            if path not in TrafficVehicle.model_collection:
                carNP = self.loader.loadModel(AssetLoader.file_path("models", path))
                TrafficVehicle.model_collection[path] = carNP
            else:
                carNP = TrafficVehicle.model_collection[path]
            carNP.setScale(scale)
            carNP.setH(H)
            carNP.setPos(x_y_z_offset)
            carNP.setZ(-self.config[Parameter.tire_radius] - 0.2)
            carNP.instanceTo(self.origin)

    def set_break_down(self, break_down=True):
        self.break_down = break_down
