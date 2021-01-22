from pgdrive.world.pg_world import PGWorld
from pgdrive.utils.math_utils import norm


class PGLOD:
    """
    Used to cull distant rendering object to improve rendering efficiency
    TODO calculation efficiency can also be improved in the future
    """
    # Visualization cull
    LOD_MAP_VIS_DIST = 300  # highly related to the render efficiency !
    LOD_VEHICLE_VIS_DIST = 500

    # Physics world cull, which can save the time used to do collision detection
    LOD_MAP_PHYSICS_DIST = 50
    LOD_VEHICLE_PHYSICS_DIST = 50

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
    def cull_distant_traffic_vehicles(cls, vehicles: list, pos, pg_world: PGWorld):
        # Cull distant vehicles
        for v in vehicles:
            v_p = v.position
            if norm(v_p[0] - pos[0], v_p[1] - v_p[1]) < cls.LOD_VEHICLE_VIS_DIST:
                if not v.node_path.hasParent():
                    v.node_path.reparentTo(pg_world.pbr_worldNP)
            else:
                if v.node_path.hasParent():
                    v.node_path.detachNode()

            if norm(v_p[0] - pos[0], v_p[1] - v_p[1]) < cls.LOD_VEHICLE_PHYSICS_DIST:
                v.dynamic_nodes.attach_to_physics_world(pg_world.physics_world.dynamic_world)
            else:
                v.dynamic_nodes.detach_from_physics_world(pg_world.physics_world.dynamic_world)
