import logging

try:
    import geopandas as gpd
except ImportError:
    pass

import numpy as np
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from metadrive.utils.coordinates_shift import nuplan_to_metadrive_vector
from metadrive.component.lane.point_lane import PointLane

from metadrive.utils.interpolating_line import InterpolatingLine
logger = logging.getLogger(__name__)


class NuPlanLane(PointLane):
    def __init__(self, lane_meta_data, nuplan_center, need_lane_localization=False):
        """
        Extract the lane information of one waymo lane, and do coordinate shift
        """

        if isinstance(lane_meta_data.polygon.boundary, MultiLineString):
            # logger.warning("Stop using boundaries! Use exterior instead!")
            boundary = gpd.GeoSeries(lane_meta_data.polygon.boundary).explode(index_parts=True)
            sizes = []
            for idx, polygon in enumerate(boundary[0]):
                sizes.append(len(polygon.xy[1]))
            points = boundary[0][np.argmax(sizes)].xy
        elif isinstance(lane_meta_data.polygon.boundary, LineString):
            points = lane_meta_data.polygon.boundary.xy
        polygon = [[points[0][i], points[1][i]] for i in range(len(points[0]))]
        # polygon += [[points[0][i], points[1][i], 0.] for i in range(len(points[0]))]
        polygon = nuplan_to_metadrive_vector(polygon, nuplan_center=[nuplan_center[0], nuplan_center[1]])
        super(NuPlanLane, self).__init__(
            self._extract_centerline(lane_meta_data, nuplan_center),
            width=None,  # we use width_at to get width
            polygon=polygon,
            need_lane_localization=need_lane_localization
        )
        self.index = lane_meta_data.id
        self.entry_lanes = lane_meta_data.incoming_edges,
        self.exit_lanes = lane_meta_data.outgoing_edges,
        self.left_lanes = lane_meta_data.adjacent_edges[0],
        self.right_lanes = lane_meta_data.adjacent_edges[-1]

        self.left_boundary = InterpolatingLine(self._get_boundary_points(lane_meta_data.left_boundary, nuplan_center))
        self.right_boundary = InterpolatingLine(self._get_boundary_points(lane_meta_data.right_boundary, nuplan_center))
        self.width = self.VIS_LANE_WIDTH

    @staticmethod
    def _extract_centerline(map_obj, nuplan_center):
        path = map_obj.baseline_path.discrete_path
        points = np.array([nuplan_to_metadrive_vector([pose.x, pose.y], nuplan_center) for pose in path])
        return points

    def width_at(self, longitudinal: float) -> float:
        l_pos = self.left_boundary.position(longitudinal, 0)
        r_pos = self.right_boundary.position(longitudinal, 0)
        return min(np.linalg.norm(r_pos - l_pos), self.VIS_LANE_WIDTH)

    def __del__(self):
        logging.debug("NuPlanLane is released")

    def destroy(self):
        self.index = None
        super(NuPlanLane, self).destroy()

    @staticmethod
    def _get_boundary_points(boundary, center):
        path = boundary.discrete_path
        points = np.array([nuplan_to_metadrive_vector([pose.x, pose.y], nuplan_center=center) for pose in path])
        return points


if __name__ == "__main__":
    raise ValueError("Can not be run")
    file_path = AssetLoader.file_path("waymo", "test.pkl", return_raw_style=False)
    data = read_waymo_data(file_path)
    print(data)
    lane = NuPlanLane(108, data["map"])
