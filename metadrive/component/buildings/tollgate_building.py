from metadrive.component.buildings.base_building import BaseBuilding
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.scene_utils import generate_invisible_static_wall


class TollGateBuilding(BaseBuilding):
    BUILDING_LENGTH = 10
    BUILDING_HEIGHT = 5
    MASS = 0

    def __init__(self, lane, position, heading_theta, random_seed):
        super(TollGateBuilding, self).__init__(lane, position, heading_theta, random_seed)
        air_wall = generate_invisible_static_wall(
            self.BUILDING_LENGTH, lane.width, self.BUILDING_HEIGHT / 2, object_id=self.id
        )
        self.add_body(air_wall)
        self.set_position(position, 0)
        self.set_heading_theta(heading_theta)

        if self.render:
            building_model = self.loader.loadModel(AssetLoader.file_path("models", "tollgate", "booth.gltf"))
            gate_model = self.loader.loadModel(AssetLoader.file_path("models", "tollgate", "gate.gltf"))
            building_model.setH(90)
            building_model.reparentTo(self.origin)
            gate_model.reparentTo(self.origin)

    @property
    def top_down_length(self):
        return self.BUILDING_LENGTH

    @property
    def top_down_width(self):
        return 3
