"""

RenderPipeline

Copyright (c) 2014-2016 tobspr <tobias.springer1@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

"""

from __future__ import print_function

from metadrive.render_pipeline.rplibs.six import iteritems, string_types
from metadrive.render_pipeline.rpcore.rpobject import RPObject

__all__ = ["make_setting_from_data"]


def make_setting_from_factory(data, factory):
    """ Constructs a new setting from a given dataset, alongst with a factory
    to resolve the setting type to """
    if data["type"] not in factory:
        raise Exception("Unkown setting type: {}".format(data["type"]))
    try:
        instance = factory[data["type"]](data)
    except Exception:
        print("Exception occured while parsing", data)
        raise
    if data:
        raise Exception("Unparsed data left in plugin setting: {}".format(data))
    return instance


def make_setting_from_data(data):
    """ Constructs a new setting from a given dataset. This method will automatically
    instantiate a new class matching the type of the given dataset. It will fill
    all values provided by the dataset and then return the created instance """
    factory = {
        "int": IntType,
        "float": FloatType,
        "bool": BoolType,
        "enum": EnumType,
        "path": PathType,
        "power_of_two": PowerOfTwoType,
        "sample_sequence": SampleSequenceType
    }
    return make_setting_from_factory(data, factory)


class BaseType(RPObject):
    """ This is the base setting type, all setting types derive from this """
    def __init__(self, data):
        """ Fills in all settings from the given setting data """
        self.value = None
        self.type = data.pop("type")
        self.label = data.pop("label").strip()
        self.description = data.pop("description").strip()
        self.runtime = data.pop("runtime", False)
        self.shader_runtime = data.pop("shader_runtime", False)
        self.display_conditions = data.pop("display_if", {})

        RPObject.__init__(self, "psetting:{}".format(self.label))

    def set_value(self, value):
        """ Sets the value of the setting, should get overridden """
        raise NotImplementedError()

    def add_defines(self, plugin_id, setting_id, defines):
        """ Makes the value of this plugin available as a define """
        defines["{}_{}".format(plugin_id, setting_id)] = self.value

    def should_be_visible(self, settings):
        """ Evaluates whether the plugin should be visible, taking all display
        conditions into account """
        for key, val in iteritems(self.display_conditions):
            if settings[key].value != val:
                return False
        return True


class TemplatedType(BaseType):
    """ This setting stores a single type including a minimum and maximum value.
    It is shared between integer and floating point types. """
    def __init__(self, template_type, data):
        BaseType.__init__(self, data)
        self.template_type = template_type
        self.default = template_type(data.pop("default"))
        self.value = self.default

        setting_range = data.pop("range")
        self.minval = template_type(setting_range[0])
        self.maxval = template_type(setting_range[1])

    def set_value(self, value):
        value = self.template_type(value)
        if self.minval <= value <= self.maxval:
            self.value = value
        else:
            self.error("Invalid value: {}".format(value))


class IntType(TemplatedType):
    """ Template instantiation of TemplatedType using int """
    def __init__(self, data):
        TemplatedType.__init__(self, int, data)


class PowerOfTwoType(IntType):
    """ Type for any power of two resolution """
    def __init__(self, data):
        IntType.__init__(self, data)

    def set_value(self, value):
        value = int(value)
        if self.minval <= value <= self.maxval:
            if value in [2**i for i in range(32)]:
                self.value = value
            else:
                self.error("Not a power of two: {}".format(value))
        else:
            self.error("Invalid value: {}".format(value))


class FloatType(TemplatedType):
    """ Template instantiation of TemplatedType using float """
    def __init__(self, data):
        TemplatedType.__init__(self, float, data)


class BoolType(BaseType):
    """ Boolean setting type """
    def __init__(self, data):
        BaseType.__init__(self, data)
        self.default = bool(data.pop("default"))
        self.value = self.default

    def set_value(self, value):
        if isinstance(value, string_types):
            self.value = str(value.lower()) in ("true", "1")
        else:
            self.value = bool(value)


class EnumType(BaseType):
    """ Enumeration setting type """
    def __init__(self, data):
        BaseType.__init__(self, data)
        self.values = tuple(data.pop("values"))
        self.default = data.pop("default")
        if self.default not in self.values:
            raise Exception("Enum default not in enum values: {}".format(self.default))
        self.value = self.default

    def set_value(self, value):
        if value not in self.values:
            self.error("Value '" + str(value) + "' not in enum values!")
            return
        self.value = value

    def add_defines(self, plugin_id, setting_id, defines):
        defines["{}_{}".format(plugin_id, setting_id)] = 1000 + self.values.index(self.value)

        for i, val in enumerate(self.values):
            defines["enum_{}_{}_{}".format(plugin_id, setting_id, val)] = 1000 + i


class SampleSequenceType(BaseType):
    """ Type for any 2D or 3D sample sequence """

    POISSON_2D_SIZES = (4, 8, 12, 16, 32, 64)
    POISSON_3D_SIZES = (16, 32, 64)
    HALTON_SIZES = (4, 8, 16, 32, 64, 128)

    def __init__(self, data):
        BaseType.__init__(self, data)
        self.dimension = int(data.pop("dimension"))
        if self.dimension not in (2, 3):
            raise Exception("Not a valid dimension, must be 2 or 3!")
        self.default = data.pop("default")
        if self.default not in self.sequences:
            raise Exception("Not a valid sequence: {}".format(self.default))
        self.value = self.default

    def set_value(self, value):
        if value not in self.sequences:
            self.error("Value '" + str(value) + "' is not a valid sequence!")
            return
        self.value = value

    @property
    def sequences(self):
        result = []
        if self.dimension == 2:
            for dim in self.POISSON_2D_SIZES:
                result.append("poisson_2D_" + str(dim))
        else:
            for dim in self.POISSON_3D_SIZES:
                result.append("poisson_3D_" + str(dim))
        for dim in self.HALTON_SIZES:
            result.append("halton_" + str(self.dimension) + "D_" + str(dim))
        return result


class PathType(BaseType):
    """ Path type to specify paths to files """
    def __init__(self, data):
        BaseType.__init__(self, data)
        self.default = str(data.pop("default"))
        self.value = self.default
        self.file_type = str(data.pop("file_type"))
        self.base_path = str(data.pop("base_path"))

    def set_value(self, value):
        self.value = str(value)

    def add_defines(self, *args):
        # Paths are not available in shaders
        pass
