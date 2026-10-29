from metadrive.component.buildings.base_building import BaseBuilding
from metadrive.type import MetaDriveType
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.pg.utils import generate_static_box_physics_body


class TollGateBuilding(BaseBuilding):
    BUILDING_LENGTH = 10
    BUILDING_HEIGHT = 5
    HEIGHT = BUILDING_HEIGHT
    MASS = 0

    def __init__(self, lane, position, heading_theta, random_seed):
        super(TollGateBuilding, self).__init__(position, heading_theta, lane, random_seed)
        air_wall = generate_static_box_physics_body(
            self.BUILDING_LENGTH,
            lane.width,
            self.BUILDING_HEIGHT / 2,
            object_id=self.id,
            type_name=MetaDriveType.BUILDING
        )
        self.lane_width = lane.width
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

    @property
    def WIDTH(self):
        return self.lane_width

    @property
    def LENGTH(self):
        return self.BUILDING_LENGTH
