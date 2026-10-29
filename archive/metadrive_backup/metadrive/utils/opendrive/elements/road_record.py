# -*- coding: utf-8 -*-

__author__ = "Benjamin Orthen, Stefan Urban"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["Priority Program SPP 1835 Cooperative Interacting Automobiles"]
__version__ = "1.0.2"
__maintainer__ = "Benjamin Orthen"
__email__ = "commonroad-i06@in.tum.de"
__status__ = "Released"

from abc import ABC


class RoadRecord(ABC):
    """Abstract base class to model Records (e.g. ElevationRecord) of the OpenDRIVE
    specification.

    These Records all have attributes start_pos, a, b, c, d.
    The attribute attr which is defined the RoadRecord at a given reference line position
    is calculated with the following equation:
    attr = a + b*ds + c*ds² + d*ds³
    where ds being the distance along the reference line between the start of the entry
    and the actual position.

    ds starts at zero for each RoadRecord.

    The absolute position of an elevation value is calculated by
      s = start_pos + ds



    Attributes:
      start_pos: Position in curve parameter ds where the RoadRecord starts.
      polynomial_coefficients: List of values [a, b, c, d, ...] which can be evaluated with an
        polynomial function.
    """
    def __init__(self, *polynomial_coefficients: float, start_pos: float = None):
        self.start_pos = start_pos
        self.polynomial_coefficients = []
        for coeff in polynomial_coefficients:
            self.polynomial_coefficients.append(coeff)
