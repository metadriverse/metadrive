# -*- coding: utf-8 -*-

from metadrive.utils.opendrive_map_utils.elements.road_record import RoadRecord

__author__ = "Benjamin Orthen, Stefan Urban"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["Priority Program SPP 1835 Cooperative Interacting Automobiles"]
__version__ = "1.0.2"
__maintainer__ = "Benjamin Orthen"
__email__ = "commonroad-i06@in.tum.de"
__status__ = "Released"


class ElevationProfile:
    """The elevation profile record contains a series of elevation records
    which define the characteristics of
    the road's elevation along the reference line.

    (Section 5.3.5 of OpenDRIVE 1.4)
    """
    def __init__(self):
        self.elevations = []


class ElevationRecord(RoadRecord):
    """The elevation record defines an elevation entry at a given reference line position.

    (Section 5.3.5.1 of OpenDRIVE 1.4)
    """
