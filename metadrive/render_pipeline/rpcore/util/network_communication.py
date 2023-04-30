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

import socket
from threading import Thread

from metadrive.render_pipeline.rpcore.rpobject import RPObject


class NetworkCommunication(RPObject):
    """ Listener which accepts messages on several ports to detect incoming updates.
    Also provides functionality to send updates. """

    CONFIG_PORT = 63324
    DAYTIME_PORT = 63325
    MATERIAL_PORT = 63326

    @classmethod
    def send_async(cls, port, message):
        """ Starts a new thread which sends a given message to a port """
        thread = Thread(target=cls.__send_message_async, args=(port, message), name="NC-SendAsync")
        thread.setDaemon(True)
        thread.start()
        return thread

    @classmethod
    def listen_threaded(cls, port, callback):
        """ Starts a new thread listening to the given port """
        thread = Thread(target=cls.__listen_forever, args=(port, callback), name="NC-ListenForever")
        thread.setDaemon(True)
        thread.start()
        return thread

    @staticmethod
    def __send_message_async(port, message=""):
        """ Sends a given message to a given port and immediately returns. """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(message.encode("utf-8"), ("127.0.0.1", port))
        finally:
            sock.close()

    @staticmethod
    def __listen_forever(port, callback):
        """ Listens to a given port, and calls callback in case a message
        arrives. This method never returns, except when the connection closed or
        could not be established. """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            while True:
                data, addr = sock.recvfrom(1024)  # pylint: disable=unused-variable
                callback(data.decode("utf-8"))
        finally:
            sock.close()

    def __init__(self, pipeline):
        """ Creates the listener service. This also starts listening on the various
        ports for updates """
        RPObject.__init__(self)
        self._pipeline = pipeline
        self._config_updates = set()
        self._daytime_updates = set()
        self._material_updates = set()
        self._config_thread = self.listen_threaded(self.CONFIG_PORT, self._config_updates.add)
        self._daytime_thread = self.listen_threaded(self.DAYTIME_PORT, self._daytime_updates.add)
        self._material_thread = self.listen_threaded(self.MATERIAL_PORT, self._material_updates.add)

    def update(self):
        """ Update task which gets called every frame and executes the changes.
        This takes the incoming scheduled commands and processes one at a time."""
        while self._config_updates:
            cmd = self._config_updates.pop()
            self._handle_config_command(cmd)
        while self._daytime_updates:
            cmd = self._daytime_updates.pop()
            self._handle_daytime_command(cmd)
        while self._material_updates:
            cmd = self._material_updates.pop()
            self._handle_material_command(cmd)

    def _handle_daytime_command(self, cmd):
        """ Handles a daytime command. This could either be a command to set
        the time, or a command to reload the time of day configuration. """
        if cmd.startswith("settime "):
            daytime = float(cmd.split()[1])
            self._pipeline.daytime_mgr.time = daytime
        elif cmd.startswith("loadconf"):
            self._pipeline.plugin_mgr.load_daytime_overrides("/$$rpconfig/daytime.yaml")
        else:
            self.warn("Recieved unkown daytime command:", cmd)

    def _handle_config_command(self, cmd):
        """ Handles an incomming configuration command. Currently this can only
        be an update of a plugin setting """
        if cmd.startswith("setval "):
            parts = cmd.split()
            setting_parts = parts[1].split(".")
            self._pipeline.plugin_mgr.on_setting_changed(setting_parts[0], setting_parts[1], parts[2])
        else:
            self.warn("Recieved unkown plugin command:", cmd)

    def _handle_material_command(self, cmd):
        """ Handles an incomming material command """
        if cmd.startswith("dump_materials"):
            path = cmd[len("dump_materials "):].strip()
            self.debug("Writing materials to", path)
            self._pipeline.export_materials(path)

        elif cmd.startswith("update_material"):

            data = cmd[len("update_material "):].strip()
            parts = data.split()
            self._pipeline.update_serialized_material(parts)

        else:
            self.warn("Recieved unkown plugin command:", cmd)
