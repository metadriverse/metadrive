from panda3d.core import *
from direct.directbase import DirectStart
import random


def makeGrassBlades():
    format = GeomVertexFormat.getV3n3cpt2()
    vdata = GeomVertexData('blade', format, Geom.UHStatic)
    vertex = GeomVertexWriter(vdata, 'vertex')
    normal = GeomVertexWriter(vdata, 'normal')
    texcoord = GeomVertexWriter(vdata, 'texcoord')
    blade = Geom(vdata)

    for x in range(20):
        for y in range(20):
            r = random.uniform(0, 0.4)
            vertex.addData3f((x * 0.4) - 0.0291534 + r, (y * 0.4) + 0.0101984 + r, 0.0445018)
            vertex.addData3f((x * 0.4) + 0.0338934 + r, (y * 0.4) + 0.041644 + r, 0.83197)
            vertex.addData3f((x * 0.4) + 0.0304494 + r, (y * 0.4) - 0.00795362 + r, 0.360315)
            vertex.addData3f((x * 0.4) - 0.0432457 + r, (y * 0.4) - 0.0362444 + r, 0.0416673)
            vertex.addData3f((x * 0.4) - 0.0291534 + r, (y * 0.4) + 0.0101984 + r, 0.0445018)
            normal.addData3f(0.493197, 0.854242, -0.164399)
            normal.addData3f(-0.859338, 0.496139, -0.124035)
            normal.addData3f(-0.759642, -0.637797, -0.127114)
            normal.addData3f(0.974147, -0.0584713, -0.218218)
            normal.addData3f(0.493197, 0.854242, -0.164399)
            texcoord.addData2f(0.0478854, 0.000499576)
            texcoord.addData2f(0.353887, 0.9995)
            texcoord.addData2f(0.999501, 0.363477)
            texcoord.addData2f(0.729119, 0.000499576)
            texcoord.addData2f(0.000499547, 0.000499576)

    for z in range(0, 2000, 5):
        triangles = GeomTriangles(Geom.UHStatic)
        triangles.addVertices(0 + z, 1 + z, 2 + z)
        triangles.addVertices(2 + z, 3 + z, 0 + z)
        triangles.addVertices(1 + z, 4 + z, 2 + z)
        triangles.addVertices(3 + z, 2 + z, 4 + z)
        blade.addPrimitive(triangles)

    snode = GeomNode('node')
    snode.addGeom(blade)
    return snode


for x in range(4):
    for y in range(4):
        grass_group = render.attachNewNode(makeGrassBlades())
        # grass_group.setTexture(loader.loadTexture("grass_mini.png"))
        grass_group.setColor(0, 1, 0, 1)
        grass_group.setPos(x * 8, y * 8, 0)

# grass_group.analyze()

# Create some lighting
ambientLight = AmbientLight("ambientLight")
ambientLight.setColor(Vec4(.3, .3, .3, 1))
directionalLight = DirectionalLight("directionalLight")
directionalLight.setDirection(Vec3(-5, -5, -5))
directionalLight.setColor(Vec4(1, 1, 1, 1))
directionalLight.setSpecularColor(Vec4(1, 1, 1, 1))
render.setLight(render.attachNewNode(directionalLight))
render.setLight(render.attachNewNode(ambientLight))
render.setShaderAuto()

run()
