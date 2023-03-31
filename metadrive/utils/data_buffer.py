from collections import deque

from metadrive.engine.engine_utils import get_engine
import numpy as np


def _clear_if_necessary(obj, depth=0):
    if depth > 5:
        obj = None
        del obj
        return

    if isinstance(obj, dict):
        keys = set(obj.keys())
        for k in keys:
            if k in obj:
                _clear_if_necessary(obj[k], depth + 1)
                obj[k] = None
        obj.clear()
    elif isinstance(obj, str):
        del obj
    elif isinstance(obj, list):
        obj.clear()
        del obj
    elif isinstance(obj, np.ndarray):
        del obj
    else:
        del obj


class DataBuffer:
    """
    This class builds a cache for data (mainly for Waymo data).
    You can store map / traffic tracks in it.
    The maximum size of the buffer is determined by store_data_buffer_size
    """
    def __init__(self, store_data_buffer_size=None):

        self.store_data_buffer = {}

        if store_data_buffer_size is None:
            store_data_buffer_size = 1

        self.store_data_indices = deque(maxlen=store_data_buffer_size)
        self.store_data_buffer_size = store_data_buffer_size

    def clear_if_necessary(self):
        while len(self.store_data_buffer) >= self.store_data_buffer_size:

            tmp_index = self.store_data_indices.popleft()

            if not isinstance(self.store_data_buffer[tmp_index], dict):
                self.store_data_indices.appendleft(tmp_index)
                return

            tmp_obj = self.store_data_buffer.pop(tmp_index)
            get_engine().clear_object_if_possible(tmp_obj, force_destroy=True)
            if hasattr(tmp_obj, "destroy"):
                print("Destroy object: ", type(tmp_obj))
                tmp_obj.destroy()

            _clear_if_necessary(tmp_obj)

    def __getitem__(self, item):
        assert item in self.store_data_buffer
        return self.store_data_buffer[item]

    def __contains__(self, key):
        return key in self.store_data_buffer

    def __setitem__(self, key, value):
        # This should be checked in map manager instead of here
        # map_num = get_engine().global_config["num_scenarios"]
        # start = get_engine().global_config["start_scenario_index"]
        #
        # assert start <= key < start + map_num, (start, key, start + map_num)

        self.clear_if_necessary()

        self.store_data_buffer[key] = value
        self.store_data_indices.append(key)

        # assert len(self.store_data_indices) == len(self.store_data_buffer)
        # assert len(self.store_data_indices) <= self.store_data_buffer_size
