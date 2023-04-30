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

# This script downloads and updates the module builder.

# pylint: skip-file

if __name__ == "__main__":
    import os
    import sys
    from os.path import dirname, realpath, isfile

    # Files which are skipped from the downloaded zip
    files_to_ignore = "__init__.py LICENSE README.md config.ini config_module".split()

    current_dir = dirname(realpath(__file__))
    os.chdir(current_dir)

    # Import thirdparty packages
    sys.path.insert(0, "../../rpcore/util")
    sys.path.insert(0, "../../")

    # Download the module
    from submodule_downloader import download_submodule
    download_submodule("tobspr", "P3DModuleBuilder", current_dir, files_to_ignore)

    # Make an init script so we can import the code
    with open("scripts/__init__.py", "w") as handle:
        handle.write("")

    # Update the gitignore using the suggested version from the module builder
    if isfile(".gitignore"):
        os.remove(".gitignore")
    os.rename("prefab.gitignore", ".gitignore")
