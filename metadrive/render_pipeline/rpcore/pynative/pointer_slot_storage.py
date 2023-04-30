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

from metadrive.render_pipeline.rplibs.six.moves import range  # pylint: disable=import-error


class PointerSlotStorage(object):
    """ Please refer to the native C++ implementation for docstrings and comments.
    This is just the python implementation, which does not contain documentation! """
    def __init__(self, max_size):
        self._data = [None] * max_size
        self._max_index = 0
        self._num_entries = 0

    def get_max_index(self):
        return self._max_index

    def get_num_entries(self):
        return self._num_entries

    def find_slot(self):
        # Notice: returns None in case of no free slot, and the slot otherwise, this
        # is different to the C++ Module
        for i, value in enumerate(self._data):
            if not value:
                return i
        return -1

    def find_consecutive_slots(self, num_consecutive):
        if num_consecutive == 1:
            return self.find_slot()

        for i in range(len(self._data)):
            any_taken = False
            for k in range(num_consecutive):
                if self._data[i + k]:
                    any_taken = True
                    break
            if not any_taken:
                return i
        return -1

    def free_slot(self, slot):
        self._data[slot] = None
        self._num_entries -= 1
        if slot == self._max_index:
            while self._max_index >= 0 and not self._data[self._max_index]:
                self._max_index -= 1

    def free_consecutive_slots(self, slot, num_consecutive):
        for i in range(num_consecutive):
            self.free_slot(slot + i)

    def reserve_slot(self, slot, ptr):
        self._max_index = max(self._max_index, slot)
        self._data[slot] = ptr
        self._num_entries += 1

    def begin(self):
        for i in range(self._max_index + 1):
            if self._data[i]:
                yield self._data[i]

    def end(self):
        raise NotImplementedError("Use .begin() as iterator when using the python side!")
