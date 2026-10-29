import logging
from typing import List

from metadrive.utils.math import norm

logger = logging.getLogger(__name__)


class SceneCull:
    """
    Used to cull distant rendering object in MetaDrive to improve rendering efficiency
    """
    raise DeprecationWarning
    # Visualization cull
    LOD_MAP_VIS_DIST = 300  # highly related to the render efficiency !
    LOD_VEHICLE_VIS_DIST = 500
    LOD_OBJECT_VIS_DIST = 500

    # Physics world cull, which can save the time used to do collision detection
    LOD_MAP_PHYSICS_DIST = 50
    LOD_VEHICLE_PHYSICS_DIST = 100000
    LOD_OBJECT_PHYSICS_DIST = 100000

    @classmethod
    def cull_distant_blocks(cls, engine, blocks: list, poses: List[tuple], max_distance=None):
        # A distance based LOD rendering
        for block in blocks:
            if not cls.all_out_of_bounding_box(block.bounding_box, poses, cls.LOD_MAP_VIS_DIST):
                if not block.origin.hasParent():
                    block.origin.reparentTo(engine.worldNP)
            else:
                if block.origin.hasParent():
                    block.origin.detachNode()
            if not cls.all_out_of_bounding_box(block.bounding_box, poses, max_distance or cls.LOD_MAP_PHYSICS_DIST):
                block.dynamic_nodes.attach_to_physics_world(engine.physics_world.dynamic_world)
            else:
                block.dynamic_nodes.detach_from_physics_world(engine.physics_world.dynamic_world)

    @classmethod
    def cull_distant_traffic_vehicles(cls, engine, vehicles: list, poses: List[tuple], max_distance=None):
        cls._cull_elements(
            engine, vehicles, poses, cls.LOD_VEHICLE_VIS_DIST, max_distance or cls.LOD_VEHICLE_PHYSICS_DIST
        )

    @classmethod
    def cull_distant_objects(cls, engine, objects: list, poses: List[tuple], max_distance=None):
        cls._cull_elements(engine, objects, poses, cls.LOD_OBJECT_VIS_DIST, max_distance or cls.LOD_OBJECT_PHYSICS_DIST)

    @classmethod
    def _cull_elements(cls, engine, elements: list, poses: List[tuple], vis_distance: float, physics_distance: float):
        for obj in elements:
            v_p = obj.position
            if not cls.all_distance_greater_than(vis_distance, poses, v_p):
                if not obj.origin.hasParent():
                    obj.origin.reparentTo(engine.worldNP)
            else:
                if obj.origin.hasParent():
                    obj.origin.detachNode()

            if not cls.all_distance_greater_than(physics_distance, poses, v_p):
                obj.dynamic_nodes.attach_to_physics_world(engine.physics_world.dynamic_world)
            else:
                obj.dynamic_nodes.detach_from_physics_world(engine.physics_world.dynamic_world)

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

    @staticmethod
    def out_of_bounding_box(bounding_box, pose, margin_distance):
        bb2 = [
            pose[0] - margin_distance, pose[0] + margin_distance, pose[1] - margin_distance, pose[1] + margin_distance
        ]
        if bounding_box[0] > bb2[1] or \
                bounding_box[1] < bb2[0] or \
                bounding_box[2] > bb2[3] or \
                bounding_box[3] < bb2[2]:
            return True
        else:
            return False

    @staticmethod
    def get_iou(bb1, bb2):
        raise ValueError("not checked")
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
        """
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    @staticmethod
    def no_overlap(bb1, bb2):
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return True
        else:
            return False
