from direct.directtools.DirectGeometry import LineNodePath


class MyLineNodePath(LineNodePath):
    def drawLines(self, lineList, colorList=None):
        """
        Given a list of lists of points, draw a separate line for each list
        Note: it is a list of list! a list of lines. Each line is a set of points
        The number of points in lineList[0] - 1 should equal to the number of color segments of colorList[0]
        """
        if colorList is None:
            super(MyLineNodePath, self).drawLines(lineList)
        else:
            for pointList, lineColor in zip(lineList, colorList):
                self.moveTo(*pointList[0])
                for point, seg_color, in zip(pointList[1:], lineColor):
                    self.setColor(seg_color)
                    self.drawTo(*point)
