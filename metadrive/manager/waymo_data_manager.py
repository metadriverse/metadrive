import os

from metadrive.manager.base_manager import BaseManager
from metadrive.utils.waymo_utils.waymo_utils import read_waymo_data
from tqdm import tqdm


class WaymoDataManager(BaseManager):
    def __init__(self):
        super(WaymoDataManager, self).__init__()
        self.directory = self.engine.global_config["waymo_data_directory"]
        self.case_num = self.engine.global_config["case_num"]
        for i in tqdm(range(self.case_num), desc="Check Data"):
            assert os.path.exists(os.path.join(self.directory, "{}.pkl".format(i))), "No Data"

    def get_case(self, i):
        file_path = os.path.join(self.directory, "{}.pkl".format(i))
        return read_waymo_data(file_path)

    def destroy(self):
        super(WaymoDataManager, self).destroy()
        self.cases = None
