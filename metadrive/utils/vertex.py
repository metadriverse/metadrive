import array
from copy import copy

import numpy as np
from panda3d.core import GeomVertexData, Geom, GeomVertexArrayFormat, GeomVertexFormat, \
    GeomVertexWriter, GeomNode, Triangulator, GeomTriangles, NodePath

from metadrive.utils import norm


def get_names(o_vdata):
    ret = []
    for array in o_vdata.getFormat().getArrays():
        for column in array.getColumns():
            name = column.getName()
            ret.append(name)
    return ret


def get_geom_with_class(geom, class_):
    """
    I don't know why. The debug of Pycharm cannot work for this func due to some cpp wrapping.
    But it works quite good.
    """
    # original property
    o_vdata = geom.modifyVertexData()
    o_numrow = o_vdata.getNumRows()
    o_format = o_vdata.getFormat()

    if "cls" not in get_names(o_vdata):
        new_format = copy(o_format)
        class_array = GeomVertexArrayFormat()
        class_array.addColumn("cls", 1, Geom.NTInt8, Geom.CIndex)
        new_format.addArray(class_array)
        class_format = GeomVertexFormat.registerFormat(new_format)
        o_vdata.setFormat(class_format)

        class_writer = GeomVertexWriter(o_vdata, "cls")

        while not class_writer.isAtEnd():
            class_writer.setData1i(class_)
    return o_vdata


def add_class_label(model, class_):
    geomNodeCollection = model.findAllMatches('**/+GeomNode')
    for nodePath in geomNodeCollection:
        geomNode = nodePath.node()
        for i in range(geomNode.getNumGeoms()):
            geom = geomNode.modifyGeom(i)
            new_vertex = get_geom_with_class(geom, class_=class_)
            geom.setVertexData(new_vertex)


def is_anticlockwise(points):
    """
    check if the polygon is anticlockwise
    Args:
        points: a list of 2D points representing the polygon!
    
    Returns: is anticlockwise or not 
    """
    sum = 0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i][:2]
        x2, y2 = points[(i + 1) % n][:2]  # The next point, wrapping around to the first
        sum += (x2 - x1) * (y2 + y1)
    return sum > 0


# def get_bounding_box(polygon_points):
#     """
#     Get bounding box of a polygon
#     """
#     min_x = min(polygon_points, key=lambda point: point[0])[0]
#     min_y = min(polygon_points, key=lambda point: point[1])[1]
#     max_x = max(polygon_points, key=lambda point: point[0])[0]
#     max_y = max(polygon_points, key=lambda point: point[1])[1]
#     return min_x, min_y, max_x, max_y


def make_polygon_model(points, height, auto_anticlockwise=True, force_anticlockwise=False, texture_scale=0.5):
    """
    Given a polygon represented by a set of 2D points in x-y plane, return a 3D model by extruding along z-axis.
    Args:
        points: a list of 2D points in anticlockwise order!
        height: height to extrude. If set to 0, it will generate a card
        force_anticlockwise: force making the points anticlockwise. It is helpful if your points has no order
        auto_anticlockwise: if the points are in clockwise order, we automatically reverse it.
        texture_scale: change the uv coordinate to set the texture scale

    Returns: panda3d.NodePath

    """
    need_side = abs(height) > 0.01
    if force_anticlockwise:
        points = sort_points_anticlockwise(points)
    elif not is_anticlockwise(points) and auto_anticlockwise:
        points = points[::-1]

    coords = points
    triangulator = Triangulator()
    values = array.array("f", [])
    back_side_values = array.array("f", [])
    vertex_data = GeomVertexData("poly", GeomVertexFormat.get_v3n3t2(), Geom.UH_static)
    p_num = len(coords)

    # texture coord
    for i, coord in enumerate(coords):
        x, y = [coord[0], coord[1]]
        # Top surface
        values.extend((x, y, 0., 0, 0, 1, x * texture_scale, y * texture_scale))

        pre_p = coords[(i + p_num - 1) % p_num]
        edge_1 = [x - pre_p[0], y - pre_p[1]]
        l_1 = norm(*edge_1)

        next_p = coords[(i + p_num + 1) % p_num]
        edge_2 = [next_p[0] - x, next_p[1] - y]
        l2 = norm(*edge_2)

        if l_1 < 1e-3 or l2 < 1e-3:
            normal = (0, 0, 1)
        else:
            edge_1 = [edge_1[0] / l_1, edge_1[1] / l_1]
            edge_2 = [edge_2[0] / l2, edge_2[1] / l2]

            normal = np.array([[-edge_1[1], edge_1[0], 0], [-edge_2[1], edge_2[0], 0]])
            normal = np.mean(normal, axis=0)
            normal = normal / np.linalg.norm(normal)

        if need_side:
            back_side_values.extend((x, y, -height, *normal, x * texture_scale, y * texture_scale))
        triangulator.add_vertex(x, y)
        triangulator.add_polygon_vertex(i)

    triangulator.triangulate()
    prim = GeomTriangles(Geom.UH_static)

    # add top surface
    for i in range(triangulator.get_num_triangles()):
        index0 = triangulator.get_triangle_v0(i)
        index1 = triangulator.get_triangle_v1(i)
        index2 = triangulator.get_triangle_v2(i)
        prim.add_vertices(index0, index1, index2)
        # prim.closePrimitive()

    if need_side:
        # Add side
        for i in range(p_num):
            # First triangle
            prim.add_vertices((i + 1) % p_num + p_num, i + p_num, i)
            # Second triangle
            prim.add_vertices((i + 1) % p_num + p_num, i, (i + 1) % p_num)
            prim.closePrimitive()

    # add the values to the vertex table using a memoryview;
    # since the size of a memoryview cannot change, the vertex data table
    # already needs to have the right amount of rows before creating
    # memoryviews from its array(s)
    vertex_data.unclean_set_num_rows(len(coords) * 2 if need_side else len(coords))
    # retrieve the data array for modification
    data_array = vertex_data.modify_array(0)
    memview = memoryview(data_array).cast("B").cast("f")
    if need_side:
        memview[:] = values + back_side_values
    else:
        memview[:] = values

    geom = Geom(vertex_data)
    geom.add_primitive(prim)
    node = GeomNode("polygon_node")
    node.add_geom(geom)
    return NodePath(node)


def sort_points_anticlockwise(points):
    """
    Make points anticlockwise! For now, it only works for clockwise polygon!
    Args:
        points: list of 2D point

    Returns: points ordered in anticlockwise

    """
    points = np.array(points)
    centroid = points.mean(axis=0)

    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sort_order = angles.argsort()
    return points[sort_order][::-1]
