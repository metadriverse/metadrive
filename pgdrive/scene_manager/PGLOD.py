from pgdrive.utils.math_utils import norm
from pgdrive.world.pg_world import PGWorld


class PGLOD:
    """
    Used to cull distant rendering object to improve rendering efficiency
    TODO calculation efficiency can also be improved in the future
    """
    # Visualization cull
    LOD_MAP_VIS_DIST = 300  # highly related to the render efficiency !
    LOD_VEHICLE_VIS_DIST = 500
    LOD_OBJECT_VIS_DIST = 500

    # Physics world cull, which can save the time used to do collision detection
    LOD_MAP_PHYSICS_DIST = 50
    LOD_VEHICLE_PHYSICS_DIST = 50
    LOD_OBJECT_PHYSICS_DIST = 50

    @classmethod
    def cull_distant_blocks(cls, blocks: list, pos, pg_world: PGWorld):
        # A distance based LOD rendering like GTA
        for block in blocks:
            if block.bounding_box[0] - cls.LOD_MAP_VIS_DIST < pos[0] < block.bounding_box[1] + cls.LOD_MAP_VIS_DIST and \
                    block.bounding_box[2] - cls.LOD_MAP_VIS_DIST < \
                    pos[1] < block.bounding_box[3] + cls.LOD_MAP_VIS_DIST:
                if not block.node_path.hasParent():
                    block.node_path.reparentTo(pg_world.worldNP)
            else:
                if block.node_path.hasParent():
                    block.node_path.detachNode()
            if block.bounding_box[0] - cls.LOD_MAP_PHYSICS_DIST < pos[0] < block.bounding_box[
                1] + cls.LOD_MAP_PHYSICS_DIST and \
                    block.bounding_box[2] - cls.LOD_MAP_PHYSICS_DIST < \
                    pos[1] < block.bounding_box[3] + cls.LOD_MAP_PHYSICS_DIST:
                block.dynamic_nodes.attach_to_physics_world(pg_world.physics_world.dynamic_world)
            else:
                block.dynamic_nodes.detach_from_physics_world(pg_world.physics_world.dynamic_world)

    @classmethod
    def cull_distant_traffic_vehicles(cls, vehicles: list, pos: tuple, pg_world: PGWorld):
        cls._cull_elements(vehicles, pos, pg_world, cls.LOD_VEHICLE_VIS_DIST, cls.LOD_VEHICLE_PHYSICS_DIST)

    @classmethod
    def cull_distant_objects(cls, objects: list, pos, pg_world: PGWorld):
        cls._cull_elements(objects, pos, pg_world, cls.LOD_OBJECT_VIS_DIST, cls.LOD_OBJECT_PHYSICS_DIST)

    @staticmethod
    def _cull_elements(elements: list, pos: tuple, pg_world: PGWorld, vis_distance: float, physics_distance: float):
        for obj in elements:
            v_p = obj.position
            if norm(v_p[0] - pos[0], v_p[1] - v_p[1]) < vis_distance:
                if not obj.node_path.hasParent():
                    obj.node_path.reparentTo(pg_world.pbr_worldNP)
            else:
                if obj.node_path.hasParent():
                    obj.node_path.detachNode()

            if norm(v_p[0] - pos[0], v_p[1] - v_p[1]) < physics_distance:
                obj.dynamic_nodes.attach_to_physics_world(pg_world.physics_world.dynamic_world)
            else:
                obj.dynamic_nodes.detach_from_physics_world(pg_world.physics_world.dynamic_world)
