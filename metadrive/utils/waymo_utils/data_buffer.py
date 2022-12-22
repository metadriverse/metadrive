from collections import deque

from metadrive.engine.engine_utils import get_engine


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

    def __getitem__(self, item):
        assert item in self.store_data_buffer
        return self.store_data_buffer[item]

    def __contains__(self, key):
        return key in self.store_data_buffer

    def __setitem__(self, key, value):
        engine = get_engine()

        map_num = engine.global_config["case_num"]
        start = engine.global_config["start_case_index"]

        assert start <= key < start + map_num, (start, key, start + map_num)

        while len(self.store_data_buffer) >= self.store_data_buffer_size:
            # print("[MapManager] Existing cases {} / {} exceeds the max len {}".format(
            #     len(self.maps), len(self.loaded_case), self.max_len
            # ))

            tmp_index = self.store_data_indices.popleft()
            tmp_obj = self.store_data_buffer.pop(tmp_index)

            import gc
            gc.collect()
            count = gc.get_referrers(tmp_obj)
            print("GC: ", len(count))

            engine.clear_object_if_possible(tmp_obj, force_destroy=True)
            if hasattr(tmp_obj, "destroy"):
                tmp_obj.destroy()
            del tmp_obj

        self.store_data_buffer[key] = value
        self.store_data_indices.append(key)

        assert len(self.store_data_indices) == len(self.store_data_buffer)
        assert len(self.store_data_indices) <= self.store_data_buffer_size
