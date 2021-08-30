from typing import Dict, Union

from metadrive.utils.config import Config


class Configurable:
    """
    Instances of this class will maintain a config system, which is protected from unexpected modification
    """
    def __init__(self, config: Union[Dict, Config] = None):
        # initialize and specify the value in config
        self._config = Config(config if config is not None else {})

    def get_config(self, copy=True) -> Union[Config, Dict]:
        """
        Return self._config
        :param copy:
        :return: a copy of config dict
        """
        if copy:
            return self._config.copy()
        return self._config

    def update_config(self, config: dict, allow_add_new_key=False):
        """
        Merge config and self._config
        """
        self._config.update(config, allow_add_new_key=allow_add_new_key)

    def destroy(self):
        """
        Fully delete this element and release the memory
        """
        self._config.clear()
        if hasattr(self, "engine"):
            self.engine = None

    @property
    def config(self):
        return self._config
