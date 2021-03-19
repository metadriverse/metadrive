from pgdrive.utils.math_utils import norm
from pgdrive.world.pg_world import PGWorld
from typing import List


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
    def cull_distant_blocks(cls, blocks: list, poses: List[tuple], pg_world: PGWorld):
        # A distance based LOD rendering like GTA
        for block in blocks:
            if not PGLOD.all_out_of_bounding_box(block.bounding_box, poses, cls.LOD_MAP_VIS_DIST):
                if not block.node_path.hasParent():
                    block.node_path.reparentTo(pg_world.worldNP)
            else:
                if block.node_path.hasParent():
                    block.node_path.detachNode()
            if not PGLOD.all_out_of_bounding_box(block.bounding_box, poses, cls.LOD_MAP_PHYSICS_DIST):
                block.dynamic_nodes.attach_to_physics_world(pg_world.physics_world.dynamic_world)
            else:
                block.dynamic_nodes.detach_from_physics_world(pg_world.physics_world.dynamic_world)

    @classmethod
    def cull_distant_traffic_vehicles(cls, vehicles: list, poses: List[tuple], pg_world: PGWorld):
        cls._cull_elements(vehicles, poses, pg_world, cls.LOD_VEHICLE_VIS_DIST, cls.LOD_VEHICLE_PHYSICS_DIST)

    @classmethod
    def cull_distant_objects(cls, objects: list, poses: List[tuple], pg_world: PGWorld):
        cls._cull_elements(objects, poses, pg_world, cls.LOD_OBJECT_VIS_DIST, cls.LOD_OBJECT_PHYSICS_DIST)

    @staticmethod
    def _cull_elements(
        elements: list, poses: List[tuple], pg_world: PGWorld, vis_distance: float, physics_distance: float
    ):
        for obj in elements:
            v_p = obj.position
            if not PGLOD.all_distance_greater_than(vis_distance, poses, v_p):
                if not obj.node_path.hasParent():
                    obj.node_path.reparentTo(pg_world.pbr_worldNP)
            else:
                if obj.node_path.hasParent():
                    obj.node_path.detachNode()

            if not PGLOD.all_distance_greater_than(physics_distance, poses, v_p):
                obj.dynamic_nodes.attach_to_physics_world(pg_world.physics_world.dynamic_world)
            else:
                obj.dynamic_nodes.detach_from_physics_world(pg_world.physics_world.dynamic_world)

    @staticmethod
    def all_distance_greater_than(distance, poses, target_pos):
        v_p = target_pos
        for pos in poses:
            if norm(v_p[0] - pos[0], v_p[1] - v_p[1]) < distance:
                return False
        return True

    @staticmethod
    def all_out_of_bounding_box(bounding_box, poses, margin_distance):
        for pos in poses:
            if bounding_box[0] - margin_distance < pos[0] < bounding_box[1] + margin_distance and \
                    bounding_box[2] - margin_distance < pos[1] < bounding_box[3] + margin_distance:
                return False
        return True
