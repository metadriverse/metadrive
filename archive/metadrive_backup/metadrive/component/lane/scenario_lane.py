import logging

import numpy as np

from metadrive.constants import PGDrivableAreaProperty
import math

from metadrive.component.lane.point_lane import PointLane
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.math import norm, mph_to_kmh
from metadrive.scenario.utils import read_scenario_data


# return the nearest point"s index of the line
def nearest_point(point, line):
    dist = np.square(line - point)
    dist = np.sqrt(dist[:, 0] + dist[:, 1])
    return np.argmin(dist)


class ScenarioLane(PointLane):
    VIS_LANE_WIDTH = 6
    MAX_SPEED_LIMIT = 100

    def __init__(self, lane_id: int, map_data: dict, need_lane_localization):
        """
        Extract the lane information of one lane, and do coordinate shift if required
        """
        center_line_points = np.asarray(map_data[lane_id][ScenarioDescription.POLYLINE])
        if ScenarioDescription.POLYGON in map_data[lane_id] and len(map_data[lane_id][ScenarioDescription.POLYGON]) > 3:
            polygon = np.asarray(map_data[lane_id][ScenarioDescription.POLYGON])
        else:
            polygon = None
        if "speed_limit_kmh" in map_data[lane_id] or "speed_limit_mph" in map_data[lane_id]:
            speed_limit_kmh = map_data[lane_id].get("speed_limit_kmh", None)
            if speed_limit_kmh is None:
                speed_limit_kmh = mph_to_kmh(map_data[lane_id]["speed_limit_mph"])
        else:
            speed_limit_kmh = self.MAX_SPEED_LIMIT
        super(ScenarioLane, self).__init__(
            center_line_points=center_line_points,
            width=self.VIS_LANE_WIDTH,
            polygon=polygon,
            speed_limit=speed_limit_kmh,
            need_lane_localization=need_lane_localization,
            metadrive_type=map_data[lane_id][ScenarioDescription.TYPE]
        )
        self.index = lane_id
        self.lane_type = map_data[lane_id]["type"]
        self.entry_lanes = map_data[lane_id].get(ScenarioDescription.ENTRY, None)
        self.exit_lanes = map_data[lane_id].get(ScenarioDescription.EXIT, None)
        self.left_lanes = map_data[lane_id].get(ScenarioDescription.LEFT_NEIGHBORS, None)
        self.right_lanes = map_data[lane_id].get(ScenarioDescription.RIGHT_NEIGHBORS, None)

    @staticmethod
    def try_get_polygon(map_data, lane_id):
        """
        This method tries to infer polygon from the boundaries
        """
        raise DeprecationWarning("This function can not infer the polygon still")
        lane_info = map_data[lane_id]
        center_line = lane_info["polyline"]
        left_boundary_points = []
        for boundary_info in lane_info[ScenarioDescription.LEFT_BOUNDARIES]:
            boundary = map_data[boundary_info["boundary_feature_id"]][ScenarioDescription.POLYLINE]
            start = center_line[int(boundary_info["lane_start_index"])]
            end = center_line[int(boundary_info["lane_end_index"])]

            start_on_boundary = nearest_point(start, boundary)
            end_on_boundary = nearest_point(end, boundary)

            points = boundary[start_on_boundary:end_on_boundary]
            left_boundary_points.append(points)

        left_boundary_points = np.concatenate(left_boundary_points, axis=0) if len(left_boundary_points) > 0 else []

        right_boundary_points = []
        for boundary_info in lane_info[ScenarioDescription.RIGHT_BOUNDARIES]:
            boundary = map_data[boundary_info["boundary_feature_id"]][ScenarioDescription.POLYLINE]
            start = center_line[int(boundary_info["lane_start_index"])]
            end = center_line[int(boundary_info["lane_end_index"])]

            start_on_boundary = nearest_point(start, boundary)
            end_on_boundary = nearest_point(end, boundary)

            points = boundary[start_on_boundary:end_on_boundary]
            right_boundary_points.append(points)

        right_boundary_points = np.concatenate(right_boundary_points, axis=0) if len(right_boundary_points) > 0 else []
        if len(left_boundary_points) == 0 or len(right_boundary_points) == 0:
            return None
        else:
            polygon = np.concatenate([left_boundary_points, right_boundary_points], axis=0)[..., :2]
            return np.asarray(polygon)

    def get_lane_width(self, lane_id, map_data):
        """
        We use this function to get possible lane width from raw data
        """
        return self.VIS_LANE_WIDTH
        if not (ScenarioDescription.RIGHT_NEIGHBORS in map_data[lane_id]
                and ScenarioDescription.LEFT_NEIGHBORS in map_data[lane_id]):
            return self.VIS_LANE_WIDTH
        right_lanes = map_data[lane_id][ScenarioDescription.RIGHT_NEIGHBORS]
        left_lanes = map_data[lane_id][ScenarioDescription.LEFT_NEIGHBORS]
        if len(right_lanes) + len(left_lanes) == 0:
            return max(sum(map_data[lane_id].get("width", 0)), self.VIS_LANE_WIDTH)
        dist_to_left_lane = 0
        dist_to_right_lane = 0
        if len(right_lanes) > 0 and "feature_id" in right_lanes[0]:
            right_lane = map_data[right_lanes[0]["feature_id"]]
            self_start = int(right_lanes[0]["self_start_index"])
            neighbor_start = int(right_lanes[0]["neighbor_start_index"])
            n_point = right_lane[ScenarioDescription.POLYLINE][neighbor_start]
            self_point = map_data[lane_id][ScenarioDescription.POLYLINE][self_start]
            dist_to_right_lane = norm(n_point[0] - self_point[0], n_point[1] - self_point[1])
        if len(left_lanes) > 0 and "feature_id" in left_lanes[0]:
            left_lane = map_data[left_lanes[-1]["feature_id"]]
            self_start = int(left_lanes[-1]["self_start_index"])
            neighbor_start = int(left_lanes[-1]["neighbor_start_index"])
            n_point = left_lane[ScenarioDescription.POLYLINE][neighbor_start]
            self_point = map_data[lane_id][ScenarioDescription.POLYLINE][self_start]
            dist_to_left_lane = norm(n_point[0] - self_point[0], n_point[1] - self_point[1])
        return max(dist_to_left_lane, dist_to_right_lane, self.VIS_LANE_WIDTH)

    def __del__(self):
        logging.debug("ScenarioLane is released")

    def destroy(self):
        self.index = None
        self.entry_lanes = None
        self.exit_lanes = None
        self.left_lanes = None
        self.right_lanes = None
        super(ScenarioLane, self).destroy()


if __name__ == "__main__":
    file_path = AssetLoader.file_path("waymo", "test.pkl", unix_style=False)
    data = read_scenario_data(file_path)
    print(data)
    lane = ScenarioLane(108, data["map_features"])
