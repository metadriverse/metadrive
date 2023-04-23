# -*- coding: utf-8 -*-

__author__ = "Benjamin Orthen, Stefan Urban"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["Priority Program SPP 1835 Cooperative Interacting Automobiles"]
__version__ = "1.0.2"
__maintainer__ = "Benjamin Orthen"
__email__ = "commonroad-i06@in.tum.de"
__status__ = "Released"


class OpenDrive:
    """ """
    def __init__(self):
        self.header = None
        self._roads = []
        self._controllers = []
        self._junctions = []
        self._junctionGroups = []
        self._stations = []

    # @property
    # def header(self):
    #     return self._header

    @property
    def roads(self):
        """ """
        return self._roads

    def getRoad(self, id_):
        """

        Args:
          id_: 

        Returns:

        """
        for road in self._roads:
            if road.id == id_:
                return road

        return None

    @property
    def controllers(self):
        """ """
        return self._controllers

    @property
    def junctions(self):
        """ """
        return self._junctions

    def getJunction(self, junctionId):
        """

        Args:
          junctionId: 

        Returns:

        """
        for junction in self._junctions:
            if junction.id == junctionId:
                return junction
        return None

    @property
    def junctionGroups(self):
        """ """
        return self._junctionGroups

    @property
    def stations(self):
        """ """
        return self._stations


class Header:
    """ """
    def __init__(
        self,
        rev_major=None,
        rev_minor=None,
        name: str = None,
        version=None,
        date=None,
        north=None,
        south=None,
        east=None,
        west=None,
        vendor=None,
    ):
        self.revMajor = rev_major
        self.revMinor = rev_minor
        self.name = name
        self.version = version
        self.date = date
        self.north = north
        self.south = south
        self.east = east
        self.west = west
        self.vendor = vendor
