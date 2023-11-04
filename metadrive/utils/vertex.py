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


def make_polygon_model(points, height, force_anticlockwise=False):
    """
    Given a polygon represented by a set of 2D points in x-y plane, return a 3D model by extruding along z-axis.
    Args:
        points: a list of 2D points
        height: height to extrude
        force_anticlockwise: force making the points anticlockwise. It is helpful if your points has no order

    Returns: panda3d.NodePath

    """
    coords = sort_points_anticlockwise(points) if force_anticlockwise else points
    triangulator = Triangulator()
    values = array.array("f", [])
    back_side_values = array.array("f", [])
    vertex_data = GeomVertexData("poly", GeomVertexFormat.get_v3n3t2(), Geom.UH_static)
    p_num = len(coords)

    # texture coord
    for i, coord in enumerate(coords):
        x, y = [coord[0], coord[1]]
        # Top surface
        values.extend((x, y, 0., 0, 0, 1, 0.0, 0.0))

        pre_p = coords[(i + p_num - 1) % p_num]
        edge_1 = [x - pre_p[0], y - pre_p[1]]
        l_1 = norm(*edge_1)
        edge_1 = [edge_1[0] / l_1, edge_1[1] / l_1]

        next_p = coords[(i + p_num + 1) % p_num]
        edge_2 = [next_p[0] - x, next_p[1] - y]
        l2 = norm(*edge_2)
        edge_2 = [edge_2[0] / l2, edge_2[1] / l2]

        normal = np.array([[edge_1[1], -edge_1[0], 0], [edge_2[1], -edge_2[0], 0]])
        normal = np.mean(normal, axis=0)
        normal = normal / np.linalg.norm(normal)

        back_side_values.extend((x, y, -height, *normal, 0.0, 0.0))
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
        prim.closePrimitive()

    for i in range(p_num):
        # First triangle
        prim.add_vertices(i + p_num, (i + 1) % p_num + +p_num, i)
        # Second triangle
        prim.add_vertices((i + 1) % p_num + p_num, (i + 1) % p_num, i)
        prim.closePrimitive()

    # add the values to the vertex table using a memoryview;
    # since the size of a memoryview cannot change, the vertex data table
    # already needs to have the right amount of rows before creating
    # memoryviews from its array(s)
    vertex_data.unclean_set_num_rows(len(coords) * 2)
    # retrieve the data array for modification
    data_array = vertex_data.modify_array(0)
    memview = memoryview(data_array).cast("B").cast("f")
    memview[:] = values + back_side_values

    geom = Geom(vertex_data)
    geom.add_primitive(prim)
    node = GeomNode("polygon_node")
    node.add_geom(geom)
    return NodePath(node)


def sort_points_anticlockwise(points):
    """
    Make points anticlockwise!
    Args:
        points: list of 2D point

    Returns: points ordered in anticlockwise

    """
    points = np.array(points)
    centroid = points.mean(axis=0)

    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sort_order = angles.argsort()
    return points[sort_order]  # Reverse to get clockwise order
