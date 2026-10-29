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

# pylint: disable=invalid-name

# This file includes all modules from the native module.

from __future__ import print_function
import sys
from os.path import dirname, realpath

from direct.stdpy.file import join, isfile
from metadrive.render_pipeline.rpcore.rpobject import RPObject

# Store a global flag, indicating whether the C++ modules were loaded or the python
# implemetation of them
NATIVE_CXX_LOADED = True

# Read the configuration from the flag-file, LQY: Bypass this check process.
# current_path = dirname(realpath(__file__))
# cxx_flag_path = join(current_path, "use_cxx.flag")
# if not isfile(cxx_flag_path):
#     RPObject.global_error("CORE", "Could not find cxx flag, please run the setup.py!")
#     sys.exit(1)
# else:
#     with open(join(current_path, "use_cxx.flag"), "r") as handle:
#         NATIVE_CXX_LOADED = handle.read().strip() == "1"

# The native module should only be imported once, and that by the internal pipeline code
assert __package__ == "metadrive.render_pipeline.rpcore.native", "You have included the pipeline in the wrong way!"

# Classes which should get imported
classes_to_import = [
    "GPUCommand",
    "GPUCommandList",
    "ShadowManager",
    "InternalLightManager",
    "PSSMCameraRig",
    "IESDataset",
    "TagStateManager",
]

# Classes which should get imported and renamed
classes_to_import_and_rename = {"RPPointLight": "PointLight", "RPSpotLight": "SpotLight"}

native_module = None

# If the module was built, use it, otherwise use the python wrappers
# if NATIVE_CXX_LOADED:
#     try:
from panda3d import _rplight as _native_module  # pylint: disable=wrong-import-position

# RPObject.global_debug("CORE", "Using panda3d-supplied core module")
#     except ImportError:
#         # RPObject.global_debug("CORE", "Using native core module")
#         from metadrive.render_pipeline.rpcore.native import native_ as _native_module  # pylint: disable=wrong-import-position
# else:
#     from metadrive.render_pipeline.rpcore import pynative as _native_module  # pylint: disable=wrong-import-position
#     RPObject.global_debug("CORE", "Using simulated python-wrapper module")

# Import all classes
for v in classes_to_import + list(classes_to_import_and_rename.keys()):
    if hasattr(_native_module, v):
        v_name = classes_to_import_and_rename[v] if v in classes_to_import_and_rename else v
        globals()[v_name] = getattr(_native_module, v)
    else:
        print("ERROR: could not import class", v, "from", _native_module.__name__)

# Don't export all variables, only the required ones
__all__ = classes_to_import + list(classes_to_import_and_rename.values()) + ["NATIVE_CXX_LOADED"]
