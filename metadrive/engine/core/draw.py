from direct.directtools.DirectGeometry import LineNodePath
from panda3d.core import VBase4
from metadrive.constants import CamMask


class ColorLineNodePath(LineNodePath):
    def __init__(self, parent=None, thickness=1.0):
        super(ColorLineNodePath, self).__init__(parent, name=None, thickness=thickness, colorVec=VBase4(1))
        self.hide(CamMask.Shadow)

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
                    self.setColor(seg_color)
                    self.drawTo(*point)
        self.create()
