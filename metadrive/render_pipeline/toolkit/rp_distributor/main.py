""" Tool to distribute the rp and the panda3d build. """

from __future__ import print_function

import os
import sys
import shutil
from os.path import isdir, join, dirname, realpath, relpath

base_dir = realpath(dirname(__file__))
rp_dir = realpath(join(base_dir, "../../"))
os.chdir(base_dir)

sys.path.insert(0, rp_dir)
from rplibs.six.moves import input  # noqa # pylint: disable=import-error

# TODO: Add option to skip gui folders if debugger is disabled

rp_ignores = [

    # data
    "skybox-blend.zip",
    "skybox.jpg",
    "skybox-2.jpg",
    "default_cubemap/source",
    "default_cubemap/source_2",
    "default_cubemap/filter.compute.glsl",
    "default_cubemap/filter.py",
    "data/generate_txo_files.py",
    "README.md",
    "environment_brdf/generate_reference.py",
    "run_mitsuba.bat",
    ".mip",
    ".xml",
    ".exr",
    ".psd",
    ".diff",
    ".pyc",
    ".pdb",
    "__pycache__",
    "environment_brdf/res/",
    "film_grain/generate.py",
    "film_grain/grain.compute.glsl",
    "ies_profiles/PREVIEWS.jpg",
    "loading_screen_bg.png",


    # rpcore
    "native/scripts",
    "native/source",
    "native/win_amd",
    "native/win_i386",
    "native/.gitignore",
    "native/build.py",
    "native/CMakeLists",
    "native/update_module_builder.py",
    "native/config.ini",

    # rpplugins
    ".ffxml",
    "bloom/resources/SOURCE.txt",
    "bloom/resources/lens_dirt.png",
    "clouds/resources/generate_",
    "clouds/resources/noise.inc",
    "clouds/resources/precompute.py",
    "color_correction/resources/film_luts_raw",
    "color_correction/resources/generate_",
    "plugin_prefab",
    "scattering/resources/hosek_wilkie_scattering",

    "toolkit",

]

panda_ignores = [
    ".pdb",
    ".pyc",
    "python/.vs",
    "python/include",
]

app_ignores = [
    ".pyc",
    ".blend"
]


def copy_tree(source_dir, dest_dir, ignorelist, tree_pth):
    source = join(source_dir, tree_pth)
    dest = join(dest_dir, tree_pth)
    for basepath, dirnames, files in os.walk(source):
        for f in files:
            abspath = realpath(join(basepath, f))
            abspath = abspath.replace("\\", "/")

            for ignore in ignorelist:
                if ignore in abspath:
                    break
            else:
                local_pth = relpath(abspath, start=source)
                dest_pth = join(dest, local_pth)
                dname = dirname(dest_pth)
                if not isdir(dname):
                    print("Creating", dname)
                    os.makedirs(dname)
                shutil.copyfile(abspath, dest_pth)


def distribute():
    print("Render Pipeline Distributor v0.1")
    print("")
    print("Copying tree ..")

    dist_folder_name = "built"

    tmp_dir = join(base_dir, dist_folder_name, "render_pipeline")
    if isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    for dname in ("config", "data", "rplibs", "effects", "rpcore", "rpplugins", "toolkit"):
        copy_tree(rp_dir, tmp_dir, rp_ignores, dname)

    shutil.copyfile(join(rp_dir, "LICENSE.txt"), join(tmp_dir, "LICENSE.txt"))

    print("Copying Panda3D build ...")

    python_pth = realpath(dirname(sys.executable))
    panda_pth = realpath(join(python_pth, ".."))

    tmp_dir = join(base_dir, dist_folder_name, "panda3d")
    if isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    for dname in ("direct", "etc", "models", "panda3d", "pandac", "python"):
        copy_tree(panda_pth, tmp_dir, panda_ignores, dname)

    copy_tree(panda_pth, tmp_dir, panda_ignores + [".exe"], "bin")

    shutil.copyfile(join(panda_pth, "LICENSE"), join(tmp_dir, "LICENSE.txt"))

    # Copy launcher script
    shutil.copyfile(
        join(base_dir, "launch.templ.bat"), join(base_dir, dist_folder_name, "launch.bat"))

    # Copy application
    app_pth = join(base_dir, "../../../RenderPipeline-Samples/01-Material-Demo/")
    tmp_dir = join(base_dir, dist_folder_name, "application")

    copy_tree(app_pth, tmp_dir, app_ignores, ".")

if __name__ == "__main__":
    distribute()
