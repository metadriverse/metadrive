from pgdrive.scene_creator.object.base_object import BaseObject
from pgdrive.utils.engine_utils import get_pgdrive_engine


class BaseBuilding(BaseObject):
    def __init__(self, lane, lane_index, position, heading: float = 0., node_path=None):
        super(BaseBuilding, self).__init__(lane, lane_index, position, heading)
        assert node_path is not None
        self.node_path = node_path

    def destroy(self):
        engine = get_pgdrive_engine()
        self.detach_from_pg_world(engine.pg_physics_world)
        self.node_path.removeNode()
        self.dynamic_nodes.clear()
        self.static_nodes.clear()
        self._config.clear()
        self.node_path.removeNode()
