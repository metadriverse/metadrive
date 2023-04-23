# -*- coding: utf-8 -*-

import abc
import numpy as np
from metadrive.utils.opendrive.elements.eulerspiral import EulerSpiral

__author__ = "Benjamin Orthen, Stefan Urban"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["Priority Program SPP 1835 Cooperative Interacting Automobiles"]
__version__ = "1.0.2"
__maintainer__ = "Benjamin Orthen"
__email__ = "commonroad-i06@in.tum.de"
__status__ = "Released"


class Geometry(abc.ABC):
    """A road geometry record defines the layout of the road's reference
    line in the in the x/y-plane (plan view).

    The geometry information is split into a header which is common to all geometric elements.

    (Section 5.3.4.1 of OpenDRIVE 1.4)
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, start_position: float, heading: float, length: float):
        self._start_position = np.array(start_position)
        self._length = length
        self._heading = heading

    @property
    def start_position(self) -> np.array:
        """Returns the overall geometry length"""
        return self._start_position

    @property
    def length(self) -> float:
        """Returns the overall geometry length"""
        return self._length

    @property
    def heading(self) -> float:
        """Get heading of geometry.

        Returns:
          Heading, in which direction the geometry heads at start.
        """
        return self._heading

    @abc.abstractmethod
    def calc_position(self, s_pos):
        """Calculates the position of the geometry as if the starting point is (0/0)

        Args:
          s_pos:

        Returns:

        """
        return


class Line(Geometry):
    """This record describes a straight line as part of the road’s reference line.


    (Section 5.3.4.1.1 of OpenDRIVE 1.4)
    """
    def calc_position(self, s_pos):
        """

        Args:
          s_pos:

        Returns:

        """
        pos = self.start_position + np.array([s_pos * np.cos(self.heading), s_pos * np.sin(self.heading)])
        tangent = self.heading

        return (pos, tangent)


class Arc(Geometry):
    """This record describes an arc as part of the road’s reference line.


    (Section 5.3.4.1.3 of OpenDRIVE 1.4)
    """
    def __init__(self, start_position, heading, length, curvature):
        self.curvature = curvature
        super().__init__(start_position=start_position, heading=heading, length=length)

    def calc_position(self, s_pos):
        """

        Args:
          s_pos:

        Returns:

        """
        c = self.curvature
        hdg = self.heading - np.pi / 2

        a = 2 / c * np.sin(s_pos * c / 2)
        alpha = (np.pi - s_pos * c) / 2 - hdg

        dx = -1 * a * np.cos(alpha)
        dy = a * np.sin(alpha)

        pos = self.start_position + np.array([dx, dy])
        tangent = self.heading + s_pos * self.curvature

        return (pos, tangent)


class Spiral(Geometry):
    """This record describes a spiral as part of the road’s reference line.

    For this type of spiral, the curvature
    change between start and end of the element is linear.

    (Section 5.3.4.1.2 of OpenDRIVE 1.4)
    """
    def __init__(self, start_position, heading, length, curvStart, curvEnd):
        self._curvStart = curvStart
        self._curvEnd = curvEnd

        super().__init__(start_position=start_position, heading=heading, length=length)
        self._spiral = EulerSpiral.createFromLengthAndCurvature(self.length, self._curvStart, self._curvEnd)

    def calc_position(self, s_pos):
        """

        Args:
          s_pos:

        Returns:

        """
        (x, y, t) = self._spiral.calc(
            s_pos,
            self.start_position[0],
            self.start_position[1],
            self._curvStart,
            self.heading,
        )

        return (np.array([x, y]), t)


class Poly3(Geometry):
    """This record describes a cubic polynomial as part of the road’s reference line.


    (Section 5.3.4.1.4 of OpenDRIVE 1.4)
    """
    def __init__(self, start_position, heading, length, a, b, c, d):
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        super().__init__(start_position=start_position, heading=heading, length=length)

        # raise NotImplementedError()

    def calc_position(self, s_pos):
        """

        Args:
          s_pos:

        Returns:

        """
        # TODO untested

        # Calculate new point in s_pos/t coordinate system
        coeffs = [self._a, self._b, self._c, self._d]

        t = np.polynomial.polynomial.polyval(s_pos, coeffs)

        # Rotate and translate
        srot = s_pos * np.cos(self.heading) - t * np.sin(self.heading)
        trot = s_pos * np.sin(self.heading) + t * np.cos(self.heading)

        # Derivate to get heading change
        dCoeffs = coeffs[1:] * np.array(np.arange(1, len(coeffs)))
        tangent = np.polynomial.polynomial.polyval(s_pos, dCoeffs)

        return (self.start_position + np.array([srot, trot]), self.heading + tangent)


class ParamPoly3(Geometry):
    """This record describes a parametric cubic curve as part
    of the road’s reference line in a local u/v co-ordinate system.

    This record describes an arc as part of the road’s reference line.


    (Section 5.3.4.1.5 of OpenDRIVE 1.4)
    """
    def __init__(self, start_position, heading, length, aU, bU, cU, dU, aV, bV, cV, dV, pRange):
        super().__init__(start_position=start_position, heading=heading, length=length)

        self._aU = aU
        self._bU = bU
        self._cU = cU
        self._dU = dU
        self._aV = aV
        self._bV = bV
        self._cV = cV
        self._dV = dV

        if pRange is None:
            self._pRange = 1.0
        else:
            self._pRange = pRange

    def calc_position(self, s_pos):
        """

        Args:
          s_pos:

        Returns:

        """

        # Position
        pos = (s_pos / self.length) * self._pRange

        coeffsU = [self._aU, self._bU, self._cU, self._dU]
        coeffsV = [self._aV, self._bV, self._cV, self._dV]

        x = np.polynomial.polynomial.polyval(pos, coeffsU)
        y = np.polynomial.polynomial.polyval(pos, coeffsV)

        xrot = x * np.cos(self.heading) - y * np.sin(self.heading)
        yrot = x * np.sin(self.heading) + y * np.cos(self.heading)

        # Tangent is defined by derivation
        dCoeffsU = coeffsU[1:] * np.array(np.arange(1, len(coeffsU)))
        dCoeffsV = coeffsV[1:] * np.array(np.arange(1, len(coeffsV)))

        dx = np.polynomial.polynomial.polyval(pos, dCoeffsU)
        dy = np.polynomial.polynomial.polyval(pos, dCoeffsV)

        tangent = np.arctan2(dy, dx)

        return (self.start_position + np.array([xrot, yrot]), self.heading + tangent)
