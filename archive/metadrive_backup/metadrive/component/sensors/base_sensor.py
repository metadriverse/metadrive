class BaseSensor:
    """
    This is the base class of all sensors
    """
    def perceive(self, *args, **kwargs):
        """
        All sensors have to implement this API as the interface for accessing the sensor output
        Args:
            *args: varies according to sensor type
            **kwargs: varies according to sensor type

        Returns: sensor output. It could be matrices like images or other data structures

        """
        raise NotImplementedError
