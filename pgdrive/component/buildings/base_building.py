from pgdrive.component.static_object.base_static_object import BaseStaticObject
from pgdrive.engine.engine_utils import get_engine


class BaseBuilding(BaseStaticObject):
    def __init__(self, lane, lane_index, position, heading: float = 0., node_path=None, random_seed=None):
        super(BaseBuilding, self).__init__(lane, lane_index, position, heading, random_seed)
        assert node_path is not None
        self.origin = node_path

    def destroy(self):
        engine = get_engine()
        self.detach_from_world(engine.physics_world)
        self.origin.removeNode()
        self.dynamic_nodes.clear()
        self.static_nodes.clear()
        self._config.clear()
        self.origin.removeNode()
