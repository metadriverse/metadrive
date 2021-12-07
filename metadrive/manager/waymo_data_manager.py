import os

from metadrive.manager.base_manager import BaseManager
from metadrive.utils.waymo_map_utils import read_waymo_data


class WaymoDataManager(BaseManager):

    def __init__(self):
        super(WaymoDataManager, self).__init__()
        directory = self.engine.global_config["waymo_data_directory"]
        self.case_num = self.engine.global_config["case_num"]
        self.cases = {}
        for i in range(self.case_num):
            file_path = os.path.join(directory, "{}.pkl".format(i))
            data = read_waymo_data(file_path)
            self.cases[i] = data
    
    def destroy(self):
        super(WaymoDataManager, self).destroy()
        self.cases = None
