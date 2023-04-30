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

from metadrive.render_pipeline.rplibs.yaml import load_yaml_file
from metadrive.render_pipeline.rpcore.rpobject import RPObject


class TaskScheduler(RPObject):
    """ This class manages the scheduled tasks and splits them over multiple
    frames. Plugins can query whether their subtasks should be executed
    or queued for later frames. Also performs analysis on the task configuration
    to figure if tasks are distributed uniformly. """
    def __init__(self, pipeline):
        RPObject.__init__(self)
        self._pipeline = pipeline
        self._tasks = []
        self._frame_index = 0
        self._load_config()

    def _load_config(self):
        """ Loads the tasks distribution configuration """
        config = load_yaml_file("/$$rpconfig/task-scheduler.yaml")["frame_cycles"]
        for frame_name, tasks in config:  # pylint: disable=unused-variable
            self._tasks.append(tasks)

    def _check_missing_schedule(self, task_name):
        """ Checks whether the given task is scheduled at some point. This can
        be used to check whether any task is missing in the task scheduler config. """
        for tasks in self._tasks:
            if task_name in tasks:
                break
        else:
            self.error("Task '" + task_name + "' is never scheduled and thus will never run!")

    def is_scheduled(self, task_name):
        """ Returns whether a given task is supposed to run this frame """
        self._check_missing_schedule(task_name)
        return task_name in self._tasks[self._frame_index]

    def step(self):
        """ Advances one frame """
        self._frame_index = (self._frame_index + 1) % len(self._tasks)

    @property
    def num_tasks(self):
        """ Returns the total amount of tasks """
        return sum((len(i) for i in self._tasks))

    @property
    def num_scheduled_tasks(self):
        """ Returns the amount of scheduled tasks this frame """
        return len(self._tasks[self._frame_index])
