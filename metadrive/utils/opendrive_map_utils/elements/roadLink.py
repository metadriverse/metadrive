# -*- coding: utf-8 -*-
"""Provide road link classes for the OpenDRIVE implementation."""

__author__ = "Benjamin Orthen, Stefan Urban"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["Priority Program SPP 1835 Cooperative Interacting Automobiles"]
__version__ = "1.0.2"
__maintainer__ = "Benjamin Orthen"
__email__ = "commonroad-i06@in.tum.de"
__status__ = "Released"


class Link:
    """"""
    def __init__(self, link_id=None, predecessor=None, successor=None, neighbors=None):
        self.id_ = link_id
        self.predecessor = predecessor
        self.successor = successor
        self.neighbors = [] if neighbors is None else neighbors

    def __str__(self):
        return " > link id " + str(self._id) + " | successor: " + str(self._successor)

    @property
    def id_(self):
        """ """
        return self._id

    @id_.setter
    def id_(self, value):
        """

        Args:
          value:

        Returns:

        """
        # pylint: disable=W0201
        self._id = int(value) if value is not None else None

    @property
    def predecessor(self):
        """ """
        return self._predecessor

    @predecessor.setter
    def predecessor(self, value):
        """

        Args:
          value:

        Returns:

        """
        if not isinstance(value, Predecessor) and value is not None:
            raise TypeError("Value must be Predecessor")

        # pylint: disable=W0201
        self._predecessor = value

    @property
    def successor(self):
        """ """
        return self._successor

    @successor.setter
    def successor(self, value):
        """

        Args:
          value:

        Returns:

        """
        if not isinstance(value, Successor) and value is not None:
            raise TypeError("Value must be Successor")

        # pylint: disable=W0201
        self._successor = value

    @property
    def neighbors(self):
        """ """
        return self._neighbors

    @neighbors.setter
    def neighbors(self, value):
        """

        Args:
          value:

        Returns:

        """
        if not isinstance(value, list) or not all(isinstance(x, Neighbor) for x in value):
            raise TypeError("Value must be list of instances of Neighbor.")

        # pylint: disable=W0201
        self._neighbors = value

    def addNeighbor(self, value):
        """

        Args:
          value:

        Returns:

        """
        if not isinstance(value, Neighbor):
            raise TypeError("Value must be Neighbor")

        self._neighbors.append(value)


class Predecessor:
    """ """
    def __init__(self, element_type=None, element_id=None, contact_point=None):
        self.elementType = element_type
        self.element_id = element_id
        self.contactPoint = contact_point

    def __str__(self):
        return (str(self._elementType) + " with id " + str(self._elementId) + " contact at " + str(self._contactPoint))

    @property
    def elementType(self):
        """ """
        return self._elementType

    @elementType.setter
    def elementType(self, value):
        """

        Args:
          value:

        Returns:

        """
        if value not in ["road", "junction"]:
            raise AttributeError("Value must be road or junction")

        # pylint: disable=W0201
        self._elementType = value

    @property
    def element_id(self):
        """ """
        return self._elementId

    @element_id.setter
    def element_id(self, value):
        """

        Args:
          value:

        Returns:

        """
        # pylint: disable=W0201
        self._elementId = int(value)

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
        if value not in ["start", "end"] and value is not None:
            raise AttributeError("Value must be start or end")

        # pylint: disable=W0201
        self._contactPoint = value


class Successor(Predecessor):
    """ """


class Neighbor:
    """ """
    def __init__(self, side=None, element_id=None, direction=None):
        self._side = side
        self._elementId = element_id
        self._direction = direction

    @property
    def side(self):
        """ """
        return self._side

    @side.setter
    def side(self, value):
        """

        Args:
          value:

        Returns:

        """
        if value not in ["left", "right"]:
            raise AttributeError("Value must be left or right")

        # pylint: disable=W0201
        self._side = value

    @property
    def element_id(self):
        """ """
        return self._elementId

    @element_id.setter
    def element_id(self, value):
        """

        Args:
          value:

        Returns:

        """
        # pylint: disable=W0201
        self._elementId = int(value)

    @property
    def direction(self):
        """ """
        return self._direction

    @direction.setter
    def direction(self, value):
        """

        Args:
          value:

        Returns:

        """
        if value not in ["same", "opposite"]:
            raise AttributeError("Value must be same or opposite")

        self._direction = value
