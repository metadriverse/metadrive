from panda3d.core import GeomVertexReader, GeomVertexData, Geom, GeomVertexArrayFormat, GeomVertexFormat, \
    GeomVertexWriter, GeomNode
from copy import copy

from metadrive.engine.asset_loader import initialize_asset_loader, AssetLoader
from metadrive.tests.vis_block.vis_block_base import TestBlock


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
