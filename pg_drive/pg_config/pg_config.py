import numpy as np


class PgConfig:
    """
    This class aims to check whether user config exists if default config system,
    Mostly, Config is sampled from parameter space in PgDrive
    """
    def __init__(self, config: dict):
        self._config = config

    def update(self, new_dict: dict):
        own_keys = set(self._config.keys())
        new_keys = set(new_dict.keys())
        if own_keys >= new_keys:
            for key, value in new_dict.items():
                self[key] = value
        else:
            raise KeyError(
                "Unexpected keys: {} in new dict when update config. Existing keys: {}.".format(
                    new_keys - own_keys, own_keys
                )
            )

    def __getitem__(self, item):
        assert item in self._config, "KeyError: {} doesn't exist in config".format(item)
        ret = self._config[item]
        if isinstance(ret, np.ndarray) and len(ret) == 1:
            # handle 1-d box shape sample
            ret = ret[0]
        return ret

    def __setitem__(self, key, value):
        assert key in self._config, "KeyError: {} doesn't exist in config".format(key)
        if self._config[key] is not None and value is not None:
            assert isinstance(value, type(self._config[key])), "TypeError: {}:{}".format(key, value)
        if not isinstance(self._config[key], dict):
            self._config[key] = value
        else:
            self._config[key].update(value)

    def __contains__(self, item):
        return item in self._config

    def clear(self):
        self._config.clear()

    def add(self, key, value):
        assert key not in self._config, "KeyError: {} exists in config".format(key)
        self._config[key] = value
