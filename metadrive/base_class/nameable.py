import logging

from metadrive.utils import random_string


class Nameable:
    """
    Instance of this class will have a special name
    """
    def __init__(self, name=None):
        # ID for object
        self.name = random_string() if name is None else name
        self.id = self.name  # name = id

    @property
    def class_name(self):
        return self.__class__.__name__

    def __del__(self):
        try:
            str(self)
        except AttributeError:
            pass
        else:
            logging.debug("{} is destroyed".format(str(self)))

    def __repr__(self):
        return "{}".format(str(self))

    def __str__(self):
        return "{}, ID:{}".format(self.class_name, self.name)

    def rename(self, new_name):
        self.name = new_name
        self.id = self.name
