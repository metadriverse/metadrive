import shapely
import heapq
from typing import Union
from metadrive.utils.math import norm
from shapely.geometry import Polygon, LineString
from shapely.ops import split
from shapely.geometry import Polygon
from itertools import combinations


def cut_polygon_along_parallel_edges(polygon):
    """
    (Deprecated) Given a polygon, cut it into several slices. This is for creating the crosswalk.
    Args:
        polygon: a list of 2D points,

    Returns: polygon pieces, which forms the original polygon after being combined.

    """
    # Your original polygon
    # Make sure to replace this with your actual polygon
    raise DeprecationWarning("Stop using this. Not robust enough")
    polygon = Polygon(polygon)
    # try:
    ret = find_longest_parallel_edges(polygon)
    if ret is None:
        pieces = [polygon]
    else:
        edges, _ = ret
        # Identify the two parallel edges (assuming you have their coordinates)
        edge1 = LineString(edges[0])
        edge2 = LineString(edges[1])

        # Determine how many pieces you want to split the polygon into
        num_pieces = 5

        # Create cutting lines
        cutting_lines = []
        for i in range(1, num_pieces):
            # Calculate the points for the cutting line at the current split ratio
            ratio = i / num_pieces
            point1 = edge1.interpolate(ratio, normalized=True)
            point2 = edge2.interpolate(ratio, normalized=True)
            cutting_line = LineString([point1, point2])
            cutting_lines.append(cutting_line)

        # Split the polygon using the cutting lines
        pieces = [polygon]
        for cutting_line in cutting_lines:
            for piece in pieces:
                # Split each piece further
                splitted = split(piece, cutting_line)
                if len(splitted.geoms) > 1:  # If the piece was split
                    pieces.remove(piece)  # Remove the original piece
                    pieces.extend(splitted.geoms)  # Add the new pieces
    return [piece.exterior.coords for piece in pieces][::2]


def calculate_slope(p1, p2):
    """
    Calculate the slope of a line segment.
    Args:
        p1: point 1
        p2: point 2

    Returns:

    """
    # Handle the case of a vertical line segment
    if p1[0] == p2[0]:
        return float('inf')
    else:
        return (p2[1] - p1[1]) / (p2[0] - p1[0])


def length(edge):
    """
    Return the length of an edge
    Args:
        edge: edge, two points

    Returns:

    """
    p_1 = edge[0]
    p_2 = edge[1]
    return norm(p_1[0] - p_2[0], p_1[1] - p_2[1])


def size(edge):
    """
    The size of the edge
    Args:
        edge: two points defining an edge

    Returns: length^2 of a vector

    """
    p_1 = edge[0]
    p_2 = edge[1]
    x = p_1[0] - p_2[0]
    y = p_1[1] - p_2[1]
    return x**2 + y**2


def find_longest_parallel_edges(polygon: Union[shapely.geometry.Polygon, list]):
    """
    Find and return the longest parallel edges of a polygon. If it can not find, return the longest two edges instead.
    Args:
        polygon: shapely.Polygon or list of 2D points representing a polygon

    Returns:

    """

    edges = []
    longest_parallel_edges = None
    coords = list(polygon.exterior.coords) if isinstance(polygon, shapely.geometry.Polygon) else polygon

    # Extract the edges from the polygon
    for i in range(len(coords) - 1):
        edge = (coords[i], coords[i + 1])
        edges.append(edge)
    edges.append((coords[-1], coords[0]))

    # Compare each edge with every other edge
    for edge1, edge2 in combinations(edges, 2):
        slope1 = calculate_slope(*edge1)
        slope2 = calculate_slope(*edge2)

        # Check if the slopes are equal (or both are vertical)
        if abs(slope1 - slope2) < 0.5:
            max_len = max(length(edge1), length(edge2))
            if longest_parallel_edges is None or max_len > longest_parallel_edges[-1]:
                longest_parallel_edges = ((edge1, edge2), max_len)

    if longest_parallel_edges:
        return longest_parallel_edges[0]
    else:
        # return sorted(edges, key=lambda edge: size(edge))[-2:]
        return heapq.nlargest(2, edges, key=lambda edge: size(edge))


def find_longest_edge(polygon: Union[shapely.geometry.Polygon, list]):
    """
    Return the longest edge of a polygon
    Args:
        polygon: shapely.Polygon or list of 2D points representing a polygon

    Returns: the longest edge

    """
    coords = list(polygon.exterior.coords) if isinstance(polygon, shapely.geometry.Polygon) else polygon
    edges = []
    # Extract the edges from the polygon
    for i in range(len(coords) - 1):
        edge = (coords[i], coords[i + 1])
        edges.append(edge)
    edges.append((coords[-1], coords[0]))
    return heapq.nlargest(1, edges, key=lambda edge: size(edge))


if __name__ == '__main__':
    polygon = Polygon(
        [
            [356.83858017, -234.46019451], [355.12995531, -239.44667613], [358.76606674, -240.73795931],
            [360.27632766, -235.80099687], [356.83858017, -234.46019451]
        ]
    )
    parallel_edges = find_longest_parallel_edges(polygon)
    assert parallel_edges == [
        (
            ((355.12995531, -239.44667613), (358.76606674, -240.73795931)),
            ((360.27632766, -235.80099687), (356.83858017, -234.46019451))
        )
    ]
    print(parallel_edges)
