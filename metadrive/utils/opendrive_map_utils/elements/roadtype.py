# -*- coding: utf-8 -*-


class RoadType:
    """ """

    allowedTypes = [
        "unknown",
        "rural",
        "motorway",
        "town",
        "lowSpeed",
        "pedestrian",
        "bicycle",
    ]

    def __init__(self, s_pos=None, use_type=None, speed=None):
        self.start_pos = s_pos
        self.use_type = use_type
        self.speed = speed

    @property
    def start_pos(self):
        """ """
        return self._sPos

    @start_pos.setter
    def start_pos(self, value):
        """

        Args:
          value:

        Returns:

        """
        # pylint: disable=W0201
        self._sPos = float(value)

    @property
    def use_type(self):
        """ """
        return self._use_type

    @use_type.setter
    def use_type(self, value):
        """

        Args:
          value:

        Returns:

        """
        if value not in self.allowedTypes:
            raise AttributeError("Type not allowed.")
        # pylint: disable=W0201
        self._use_type = value

    @property
    def speed(self):
        """ """
        return self._speed

    @speed.setter
    def speed(self, value):
        """

        Args:
          value:

        Returns:

        """
        if not isinstance(value, Speed) and value is not None:
            raise TypeError("Value {} must be instance of Speed.".format(value))
        # pylint: disable=W0201
        self._speed = value


class Speed:
    """ """
    def __init__(self, max_speed=None, unit=None):
        self._max = max_speed
        self._unit = unit

    @property
    def max(self):
        """ """
        return self._max

    @max.setter
    def max(self, value):
        """

        Args:
          value:

        Returns:

        """
        self._max = str(value)

    @property
    def unit(self):
        """ """
        return self._unit

    @unit.setter
    def unit(self, value):
        """

        Args:
          value:

        Returns:

        """
        # TODO validate input
        self._unit = str(value)
