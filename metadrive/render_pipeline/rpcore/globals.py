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

__all__ = ("Globals",)


class Globals(object):  # pylint: disable=too-few-public-methods

    """ This class is a singleton to store globals widely used by the application.
    This is a wrapper around Panda3D's globals since ShowBase writes to __builtins__
    which is bad practice. This class also attempts to help IDEs to figure out
    where the variables come from and where they are defined. """

    __init__ = None

    @staticmethod
    def load(showbase):
        """ Loads the globals from a given showbase """
        Globals.base = showbase
        Globals.render = showbase.render
        Globals.clock = showbase.taskMgr.globalClock
        Globals.font = None
        Globals.resolution = None
        Globals.native_resolution = None
