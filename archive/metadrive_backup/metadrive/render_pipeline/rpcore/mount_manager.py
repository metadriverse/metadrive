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

import os
import atexit

from panda3d.core import Filename, VirtualFileSystem, get_model_path
from panda3d.core import VirtualFileMountRamdisk
from direct.stdpy.file import join, isdir, isfile

from metadrive.render_pipeline.rpcore.rpobject import RPObject


class MountManager(RPObject):
    """ This classes mounts the required directories for the pipeline to run.
    This is important if the pipeline is in a subdirectory for example. The
    mount manager also handles the lock, storing the current PID into a file
    named instance.pid and ensuring that there is only 1 instance of the
    pipeline running at one time. """
    def __init__(self, pipeline):
        """ Creates a new mount manager """
        RPObject.__init__(self)
        self._pipeline = pipeline
        self._base_path = self._find_basepath()
        self._lock_file = "instance.pid"
        self._model_paths = []
        self._write_path = None
        self._mounted = False
        self._do_cleanup = True
        self._config_dir = None

        self.debug("Auto-Detected base path to", self._base_path)
        atexit.register(self._on_exit_cleanup)

    @property
    def write_path(self):
        """ Returns the write path previously set with set_write_path, or None
        if no write path has been set yet. """
        return self._write_path

    @write_path.setter
    def write_path(self, pth):
        """ Set a writable directory for generated files. This can be a string
        path name or a multifile with openReadWrite(). If no pathname is set
        then the root directory is used.

        This feature is usually only used for debugging, the pipeline will dump
        all generated shaders and other temporary files to that directory.
        If you don't need this, you can use set_virtual_write_path(), which
        will create the temporary path in the VirtualFileSystem, thus not
        writing any files to disk. """
        if pth is None:
            self._write_path = None
            self._lock_file = "instance.pid"
        else:
            self._write_path = Filename.from_os_specific(pth).get_fullpath()
            self._lock_file = join(self._write_path, "instance.pid")

    @property
    def base_path(self):
        """ Returns the base path of the pipeline. This returns the path previously
        set with set_base_path, or the auto detected base path if no path was
        set yet """
        return self._base_path

    @base_path.setter
    def base_path(self, pth):
        """ Sets the path where the base shaders and models on are contained. This
        is usually the root of the rendering pipeline folder """
        self.debug("Set base path to '" + pth + "'")
        self._base_path = Filename.from_os_specific(pth).get_fullpath()

    @property
    def config_dir(self):
        """ Returns the config directory previously set with set_config_dir, or
        None if no directory was set yet """

    @config_dir.setter
    def config_dir(self, pth):
        """ Sets the path to the config directory. Usually this is the config/
        directory located in the pipeline root directory. However, if you want
        to load your own configuration files, you can specify a custom config
        directory here. Your configuration directory should contain the
        pipeline.yaml, plugins.yaml, daytime.yaml and configuration.prc.

        It is highly recommended you use the pipeline provided config files, modify
        them to your needs, and as soon as you think they are in a final version,
        copy them over. Please also notice that you should keep your config files
        up-to-date, e.g. when new configuration variables are added.

        Also, specifying a custom configuration_dir disables the functionality
        of the PluginConfigurator and DayTime editor, since they operate on the
        pipelines default config files.

        Set the directory to None to use the default directory. """
        self._config_dir = Filename.from_os_specific(pth).get_fullpath()

    @property
    def do_cleanup(self):
        """ Returns whether the mount manager will attempt to cleanup the
        generated files after the application stopped running """
        return self._do_cleanup

    @do_cleanup.setter
    def do_cleanup(self, cleanup):
        """ Sets whether to cleanup the tempfolder after the application stopped.
        This is mostly useful for debugging, to analyze the generated tempfiles
        even after the pipeline stopped running """
        self._do_cleanup = cleanup

    def get_lock(self):
        """ Checks if we are the only instance running. If there is no instance
        running, write the current PID to the instance.pid file. If the
        instance file exists, checks if the specified process still runs. This
        way only 1 instance of the pipeline can be run at one time. """

        # Check if there is a lockfile at all
        if isfile(self._lock_file):
            # Read process id from lockfile
            try:
                with open(self._lock_file, "r") as handle:
                    pid = int(handle.readline())
            except IOError as msg:
                self.error("Failed to read lockfile:", msg)
                return False

            # Check if the process is still running
            if self._is_pid_running(pid):
                self.error("Found running instance")
                return False

            # Process is not running anymore, we can write the lockfile
            self._write_lock()
            return True

        else:
            # When there is no lockfile, just create it and continue
            self._write_lock()
            return True

    def _find_basepath(self):
        """ Attempts to find the pipeline base path by looking at the location
        of this file """
        pth = os.path.abspath(join(os.path.dirname(os.path.realpath(__file__)), ".."))
        return Filename.from_os_specific(pth).get_fullpath()

    def _is_pid_running(self, pid):
        """ Checks if a pid is still running """

        # Code snippet from:
        # http://stackoverflow.com/questions/568271/how-to-check-if-there-exists-a-process-with-a-given-pid

        if os.name == 'posix':
            import errno
            if pid < 0:
                return False
            try:
                os.kill(pid, 0)
            except OSError as err:
                return err.errno == errno.EPERM
            else:
                return True
        else:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            process = kernel32.OpenProcess(0x100000, 0, pid)
            if process != 0:
                kernel32.CloseHandle(process)
                return True
            else:
                return False

    def _write_lock(self):
        """ Internal method to write the current process id to the instance.pid
        lockfile. This is used to ensure no second instance of the pipeline is
        running. """

        with open(self._lock_file, "w") as handle:
            handle.write(str(os.getpid()))

    def _try_remove(self, fname):
        """ Tries to remove the specified filename, returns either True or False
        depending if we had success or not """
        try:
            os.remove(fname)
            return True
        except (IOError, OSError):
            pass
        return False

    def _on_exit_cleanup(self):
        """ Gets called when the application exists """

        if self._do_cleanup:
            self.debug("Cleaning up ..")

            if self._write_path is not None:

                # Try removing the lockfile
                self._try_remove(self._lock_file)

                # Check for further tempfiles in the write path
                # We explicitely use os.listdir here instead of pandas listdir,
                # to work with actual paths
                for fname in os.listdir(self._write_path):
                    pth = join(self._write_path, fname)

                    # Tempfiles from the pipeline start with "$$" to distinguish
                    # them from user created files
                    if isfile(pth) and fname.startswith("$$"):
                        self._try_remove(pth)

                # Delete the write path if no files are left
                if len(os.listdir(self._write_path)) < 1:
                    try:
                        os.removedirs(self._write_path)
                    except IOError:
                        pass

    @property
    def is_mounted(self):
        """ Returns whether the MountManager was already mounted by calling
        mount() """
        return self._mounted

    def mount(self):
        """ Inits the VFS Mounts. This creates the following virtual directory
        structure, from which all files can be located:

        /$$rp/  (Mounted from the render pipeline base directory)
           + rpcore/
           + shader/
           + ...

        /$rpconfig/ (Mounted from config/, may be set by user)
           + pipeline.yaml
           + ...

        /$$rptemp/ (Either ramdisk or user specified)
            + day_time_config
            + shader_auto_config
            + ...

        /$$rpshader/ (Link to /$$rp/rpcore/shader)

         """
        self.debug("Setting up virtual filesystem")
        self._mounted = True

        def convert_path(pth):
            return Filename.from_os_specific(pth).get_fullpath()

        vfs = VirtualFileSystem.get_global_ptr()

        # Mount config dir as $$rpconf
        if self._config_dir is None:
            config_dir = convert_path(join(self._base_path, "config/"))
            self.debug("Mounting auto-detected config dir:", config_dir)
            vfs.mount(config_dir, "/$$rpconfig", 0)
        else:
            self.debug("Mounting custom config dir:", self._config_dir)
            vfs.mount(convert_path(self._config_dir), "/$$rpconfig", 0)

        # Mount directory structure
        vfs.mount(convert_path(self._base_path), "/$$rp", 0)
        vfs.mount(convert_path(join(self._base_path, "rpcore/shader")), "/$$rp/shader", 0)
        vfs.mount(convert_path(join(self._base_path, "effects")), "effects", 0)

        # Mount the pipeline temp path:
        # If no write path is specified, use a virtual ramdisk
        if self._write_path is None:
            self.debug("Mounting ramdisk as /$$rptemp")
            vfs.mount(VirtualFileMountRamdisk(), "/$$rptemp", 0)
        else:
            # In case an actual write path is specified:
            # Ensure the pipeline write path exists, and if not, create it
            if not isdir(self._write_path):
                self.debug("Creating temporary path, since it does not exist yet")
                try:
                    os.makedirs(self._write_path)
                except IOError as msg:
                    self.fatal("Failed to create temporary path:", msg)
            self.debug("Mounting", self._write_path, "as /$$rptemp")
            vfs.mount(convert_path(self._write_path), '/$$rptemp', 0)

        get_model_path().prepend_directory("/$$rp")
        get_model_path().prepend_directory("/$$rp/shader")
        get_model_path().prepend_directory("/$$rptemp")

    def unmount(self):
        """ Unmounts the VFS """
        raise NotImplementedError("TODO")
