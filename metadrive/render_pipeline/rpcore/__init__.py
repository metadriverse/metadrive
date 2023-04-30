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

# flake8: noqa

__all__ = ("RenderPipeline", "SpotLight", "PointLight")

# This file includes all classes from the pipeline which are exposed
from metadrive.render_pipeline.rpcore.render_pipeline import RenderPipeline
from metadrive.render_pipeline.rpcore.native import SpotLight, PointLight

# Polyfill a set_shader_inputs function for older versions of Panda.
from panda3d.core import NodePath
from direct.extensions_native.extension_native_helpers import Dtool_funcToMethod
from metadrive.render_pipeline.rplibs.six import iteritems

if not hasattr(NodePath, 'set_shader_inputs'):

    def set_shader_inputs(self, **inputs):
        set_shader_input = self.set_shader_input
        for args in iteritems(inputs):
            set_shader_input(*args)

    Dtool_funcToMethod(set_shader_inputs, NodePath)
    del set_shader_inputs
