# -*- coding: utf-8 -*-

from metadrive.utils.opendrive_map_utils.elements.roadPlanView import PlanView
from metadrive.utils.opendrive_map_utils.elements.roadLink import Link
from metadrive.utils.opendrive_map_utils.elements.roadLanes import Lanes
from metadrive.utils.opendrive_map_utils.elements.roadElevationProfile import (
    ElevationProfile,
)
from metadrive.utils.opendrive_map_utils.elements.roadLateralProfile import LateralProfile
from metadrive.utils.opendrive_map_utils.elements.junction import Junction


class Road:
    """ """
    def __init__(self):
        self._id = None
        self._name = None
        self._junction = None
        self._length = None

        self._header = None  # TODO
        self._link = Link()
        self._types = []
        self._planView = PlanView()
        self._elevationProfile = ElevationProfile()
        self._lateralProfile = LateralProfile()
        self._lanes = Lanes()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    @property
    def id(self):
        """ """
        return self._id

    @id.setter
    def id(self, value):
        """

        Args:
          value:

        Returns:

        """
        self._id = int(value)

    @property
    def name(self):
        """ """
        return self._name

    @name.setter
    def name(self, value):
        """

        Args:
          value:

        Returns:

        """
        self._name = str(value)

    @property
    def junction(self):
        """ """
        return self._junction

    @junction.setter
    def junction(self, value):
        """

        Args:
          value:

        Returns:

        """
        if not isinstance(value, Junction) and value is not None:
            raise TypeError("Property must be a Junction or NoneType")

        if value == -1:
            value = None

        self._junction = value

    @property
    def link(self):
        """ """
        return self._link

    @property
    def types(self):
        """ """
        return self._types

    @property
    def planView(self):
        """ """
        return self._planView

    @property
    def elevationProfile(self):
        """ """
        return self._elevationProfile

    @property
    def lateralProfile(self):
        """ """
        return self._lateralProfile

    @property
    def lanes(self):
        """ """
        return self._lanes
