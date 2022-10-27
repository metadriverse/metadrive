# -*- coding: utf-8 -*-

__author__ = "Benjamin Orthen, Stefan Urban"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["Priority Program SPP 1835 Cooperative Interacting Automobiles"]
__version__ = "1.0.2"
__maintainer__ = "Benjamin Orthen"
__email__ = "commonroad-i06@in.tum.de"
__status__ = "Released"


class Junction:
    """ """

    # TODO priority
    # TODO controller

    def __init__(self):
        self._id = None
        self._name = None
        self._connections = []

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
    def connections(self):
        """ """
        return self._connections

    def addConnection(self, connection):
        """

        Args:
          connection:

        Returns:

        """
        if not isinstance(connection, Connection):
            raise TypeError("Has to be of instance Connection")

        self._connections.append(connection)


class Connection:
    """ """
    def __init__(self):
        self._id = None
        self._incomingRoad = None
        self._connectingRoad = None
        self._contactPoint = None
        self._laneLinks = []

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
    def incomingRoad(self):
        """ """
        return self._incomingRoad

    @incomingRoad.setter
    def incomingRoad(self, value):
        """

        Args:
          value:

        Returns:

        """
        self._incomingRoad = int(value)

    @property
    def connectingRoad(self):
        """ """
        return self._connectingRoad

    @connectingRoad.setter
    def connectingRoad(self, value):
        """

        Args:
          value:

        Returns:

        """
        self._connectingRoad = int(value)

    @property
    def contactPoint(self):
        """ """
        return self._contactPoint

    @contactPoint.setter
    def contactPoint(self, value):
        """

        Args:
          value:

        Returns:

        """
        if value not in ["start", "end"]:
            raise AttributeError("Contact point can only be start or end.")

        self._contactPoint = value

    @property
    def laneLinks(self):
        """ """
        return self._laneLinks

    def addLaneLink(self, laneLink):
        """

        Args:
          laneLink:

        Returns:

        """
        if not isinstance(laneLink, LaneLink):
            raise TypeError("Has to be of instance LaneLink")

        self._laneLinks.append(laneLink)


class LaneLink:
    """ """
    def __init__(self):
        self._from = None
        self._to = None

    def __str__(self):
        return str(self._from) + " > " + str(self._to)

    @property
    def fromId(self):
        """ """
        return self._from

    @fromId.setter
    def fromId(self, value):
        """

        Args:
          value:

        Returns:

        """
        self._from = int(value)

    @property
    def toId(self):
        """ """
        return self._to

    @toId.setter
    def toId(self, value):
        """

        Args:
          value:

        Returns:

        """
        self._to = int(value)
