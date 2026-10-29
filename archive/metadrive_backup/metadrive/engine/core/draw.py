from direct.directtools.DirectGeometry import LineNodePath
from metadrive.engine.asset_loader import AssetLoader
from panda3d.core import VBase4, NodePath, Material
from metadrive.constants import CamMask
from panda3d.core import LVecBase4f


class ColorLineNodePath(LineNodePath):
    def __init__(self, parent=None, thickness=1.0):
        super(ColorLineNodePath, self).__init__(parent, name=None, thickness=thickness, colorVec=VBase4(1))
        self.hide(CamMask.Shadow | CamMask.DepthCam)
        self.clearShader()
        self.setShaderAuto()

    def draw_lines(self, lineList, colorList=None):
        """
        Given a list of lists of points, draw a separate line for each list
        Note: it is a list of list! a list of lines. Each line is a set of points
        The number of points in lineList[0] - 1 should equal to the number of color segments of colorList[0]
        """
        if colorList is None:
            super(ColorLineNodePath, self).drawLines(lineList)
        else:
            for pointList, lineColor in zip(lineList, colorList):
                self.moveTo(*pointList[0])
                for point, seg_color, in zip(pointList[1:], lineColor):
                    assert 3 <= len(seg_color) <= 4, "color vector should have 3 or 4 component, get {} instead".format(
                        len(seg_color)
                    )
                    if len(seg_color) == 4:
                        self.setColor(LVecBase4f(*seg_color))
                    else:
                        self.setColor(LVecBase4f(*seg_color, 1.0))
                    self.drawTo(*point)
        self.create()


class ColorSphereNodePath(NodePath):
    """
    It is used to draw points in the scenes for debugging
    """
    def __init__(self, parent=None, scale=1):
        super(ColorSphereNodePath, self).__init__("Point Debugger")
        scale /= 10
        self.scale = scale
        self.hide(CamMask.Shadow | CamMask.DepthCam)
        self.reparentTo(self.engine.render if parent is None else parent)
        self._existing_points = []
        self._dying_points = []
        self._engine = None

    @property
    def engine(self):
        """
        Return Engine
        """
        if self._engine is None:
            from metadrive.engine.engine_utils import get_engine
            self._engine = get_engine()
        return self._engine

    def draw_points(self, points, colors=None):
        """
        Draw a set of points with colors
        Args:
            points: a set of 3D points
            colors: a list of color for each point

        Returns: None

        """
        for k, point in enumerate(points):
            if len(self._dying_points) > 0:
                np = self._dying_points.pop()
            else:
                np = NodePath("debug_point")
                model = self.engine.loader.loadModel(AssetLoader.file_path("models", "sphere.egg"))
                model.setScale(self.scale)
                model.reparentTo(np)
            material = Material()
            if colors:
                material.setBaseColor(LVecBase4f(*colors[k]))
            else:
                material.setBaseColor(LVecBase4f(1, 1, 1, 1))
            material.setShininess(64)
            # material.setEmission((1, 1, 1, 1))
            np.setMaterial(material, True)
            np.setPos(*point)
            np.reparentTo(self)
            self._existing_points.append(np)

    def reset(self):
        """
        Clear all created points
        Returns: None

        """
        for np in self._existing_points:
            np.detachNode()
            self._dying_points.append(np)
