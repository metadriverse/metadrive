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

# Disable the xxx has no yyy member warning, pylint seems to be
# unable to figure out the Colorama properties, and throws an error
# for each property.
# pylint: disable=no-member

from __future__ import print_function

import sys

# Load and init colorama, used to color the output
from metadrive.render_pipeline.rplibs.colorama import init as init_colorama
from metadrive.render_pipeline.rplibs.colorama import Fore, Style
init_colorama()


class RPObject(object):
    """ This is the base class for every object in the render pipeline. It
    provides the functions debug, warn, error and fatal for classes which
    inherit from this object, including the name of the class when printing out
    the message. """

    _OUTPUT_LEVEL = 0
    _OUTPUT_LEVELS = ["debug", "warning", "error", "fatal"]

    @classmethod
    def set_output_level(cls, level):
        """ Sets the output level, messages with a level below will not be
        printed. E.g. if you set the output level to "error", only error and
        fatal messages will be shown.  """
        assert level in RPObject._OUTPUT_LEVELS
        RPObject._OUTPUT_LEVEL = RPObject._OUTPUT_LEVELS.index(level)

    @staticmethod
    def global_debug(context, *args, **kwargs):
        """ This method can be used from a static context to print a debug
        message. The first argument should be the name of the object / context,
        all other arguments should be the message. """
        if RPObject._OUTPUT_LEVEL > 0:
            return
        print(
            kwargs.get("color", Fore.GREEN) + "[>] " + context.ljust(25) + " " + Style.RESET_ALL + Fore.WHITE +
            ' '.join([str(i) for i in args]), Fore.RESET + Style.RESET_ALL
        )

    @staticmethod
    def global_warn(context, *args):
        """ This method can be used from a static context to print a warning.
        The first argument should be the name of the object / context, all
        other arguments should be the message. """
        if RPObject._OUTPUT_LEVEL > 1:
            return
        print(
            Fore.YELLOW + Style.BRIGHT + "[!] " + context.ljust(25) + Fore.YELLOW + Style.BRIGHT + " " +
            ' '.join([str(i) for i in args]) + Fore.RESET + Style.RESET_ALL
        )

    @staticmethod
    def global_error(context, *args):
        """ This method can be used from a static context to print an error.
        The first argument should be the name of the object / context, all
        other arguments should be the message. """
        if RPObject._OUTPUT_LEVEL > 2:
            return
        print(
            Fore.RED + Style.BRIGHT + "\n[!!!] " + context.ljust(23) + " " + ' '.join([str(i) for i in args]) + "\n" +
            Fore.RESET + Style.RESET_ALL
        )

    def __init__(self, name=None):
        """ Initiates the RPObject with a given name. The name should be
        representative about the class. If no name is given, the classname
        is used """
        if name is None:
            name = self.__class__.__name__
        self._debug_name = name
        self._debug_color = Fore.GREEN

    def _set_debug_color(self, color, style=None):
        """ Sets the color used to output debug messages """
        self._debug_color = getattr(Fore, color.upper())
        if style:
            self._debug_color += getattr(Style, style.upper())

    @property
    def debug_name(self):
        """ Returns the name of the debug object """
        return self._debug_name

    @debug_name.setter
    def debug_name(self, name):
        """ Renames this object """
        self._debug_name = name

    def debug(self, *args):
        """ Outputs a debug message, something that is not necessarry
        but provides useful information for the developer """
        self.global_debug(self._debug_name, *args, color=self._debug_color)

    def warn(self, *args):
        """ Outputs a warning message, something that failed or does
        not work, but does not prevent the program from running """
        self.global_warn(self._debug_name, *args)

    def error(self, *args):
        """ Outputs an error message, something really serious.
        Hopefully this never get's called! """
        self.global_error(self._debug_name, *args)

    def fatal(self, *args):
        """ Outputs a fatal error message, printing out the errors and then
        calling sys.exit to terminate the program. This method should be called
        when something failed so hard that the pipeline has to exit. """
        # We have to set output level to 0 here, so we can print out errors
        RPObject._OUTPUT_LEVEL = 0
        self.error(*args)
        sys.exit(0)

    def __repr__(self):
        """ Represents this object. Subclasses can override this """
        return self._debug_name
