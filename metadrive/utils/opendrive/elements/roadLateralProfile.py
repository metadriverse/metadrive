# -*- coding: utf-8 -*-

from metadrive.utils.opendrive.elements.road_record import RoadRecord

__author__ = "Benjamin Orthen, Stefan Urban"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["Priority Program SPP 1835 Cooperative Interacting Automobiles"]
__version__ = "1.0.2"
__maintainer__ = "Benjamin Orthen"
__email__ = "commonroad-i06@in.tum.de"
__status__ = "Released"


class LateralProfile:
    """The lateral profile record contains a series of superelevation
    and crossfall records which define the
    characteristics of the road surface's banking along the reference line.

    (Section 5.3.6 of OpenDRIVE 1.4)
    """
    def __init__(self):
        self._superelevations = []
        self._crossfalls = []
        self._shapes = []

    @property
    def superelevations(self):
        """The superelevations of a LateralProfile."""
        return self._superelevations

    @superelevations.setter
    def superelevations(self, value):
        if not isinstance(value, list) or not all(isinstance(x, Superelevation) for x in value):
            raise TypeError("Value must be a list of Superelevation.")

        self._superelevations = value

    @property
    def crossfalls(self):
        """Crossfalls of a LateralProfile. """
        return self._crossfalls

    @crossfalls.setter
    def crossfalls(self, value):
        if not isinstance(value, list) or not all(isinstance(x, Crossfall) for x in value):
            raise TypeError("Value must be a list of Crossfall.")

        self._crossfalls = value

    @property
    def shapes(self):
        """ """
        return self._shapes

    @shapes.setter
    def shapes(self, value):
        if not isinstance(value, list) or not all(isinstance(x, Shape) for x in value):
            raise TypeError("Value must be a list of instances of Shape.")

        self._shapes = value


class Superelevation(RoadRecord):
    """The superelevation of the road is defined as the
    road section’s roll angle around the s-axis.

    (Section 5.3.6.1 of OpenDRIVE 1.4)
    """


class Crossfall(RoadRecord):
    """The crossfall of the road is defined as the road
    surface ́s angle relative to the t-axis.

    (Section 5.3.6.2 of OpenDRIVE 1.4)
    """
    def __init__(self, *polynomial_coefficients: float, start_pos: float = None, side: str = None):
        super().__init__(*polynomial_coefficients, start_pos=start_pos)
        self.side = side

    @property
    def side(self) -> str:
        """The side of the crossfall.

        Returns:
          The side as a string.

        Note:
          Setter only allows to set the side for 'left', 'right' or 'both'.
        """
        return self._side

    @side.setter
    def side(self, value):
        if value not in ["left", "right", "both"]:
            raise TypeError("Value must be string with content 'left', 'right' or 'both'.")

        self._side = value


class Shape(RoadRecord):
    """The shape of the road is defined as the road section’s surface
    relative to the reference plane.

    This shape may be described as a series of 3 order polynomials for a given "s" station.

    The absolute position of a shape value is calculated by::

       t = start_pos_t + dt

    h_shape is the height above the reference path at a given position and
    is calculated by::

       h_shape = a + b*dt + c*dt² + d*dt³

    dt being the distance perpendicular to the reference line between the start of the
    entry and the actual position.

    (Section 5.3.6.3 of OpenDRIVE 1.4)

    """
    def __init__(
        self,
        *polynomial_coefficients: float,
        start_pos: float = None,
        start_pos_t: float = None,
    ):
        super().__init__(*polynomial_coefficients, start_pos=start_pos)
        self.start_pos_t = start_pos_t
