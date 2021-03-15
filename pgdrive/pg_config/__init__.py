import numpy as np
import copy


def _check_keys(new_config, old_config, prefix=""):
    if isinstance(new_config, PGConfig):
        new_config = new_config.get_dict()
    if isinstance(old_config, PGConfig):
        old_config = new_config.get_dict()
    assert isinstance(new_config, dict)
    assert isinstance(old_config, dict)
    own_keys = set(old_config)
    new_keys = set(new_config)
    if own_keys >= new_keys:
        return True
    else:
        raise KeyError(
            "Unexpected keys: {} in new dict{} when update config. Existing keys: {}.".format(
                new_keys - own_keys, "'s '{}'".format(prefix) if prefix else "", own_keys
            )
        )


def _recursive_check_keys(new_config, old_config, prefix=""):
    _check_keys(new_config, old_config, prefix)
    for k, v in new_config.items():
        new_prefix = prefix + "/" + k if prefix else k
        # if isinstance(v, dict):
        #     _recursive_check_keys(new_config[k], old_config[k], new_prefix)
        if isinstance(v, list):
            for new, old in zip(v, old_config[k]):
                _recursive_check_keys(new, old, new_prefix)


class PGConfig:
    """
    This class aims to check whether user config exists if default config system,
    Mostly, Config is sampled from parameter space in PGDrive

    Besides, the value type will also be checked, but sometimes the value type is not unique (maybe Union[str, int]).
    For these <key, value> items, use PGConfig["your key"] = None to init your PgConfig, then it will not implement
    type check at the first time. key "config" in map.py and key "force_fps" in world.py are good examples.
    """
    def __init__(self, config: dict):
        self._config = config
        self._types = dict()

    def update(self, new_dict: dict):
        _recursive_check_keys(new_dict, self._config)
        for key, value in new_dict.items():
            self[key] = value

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
            type_correct = isinstance(value, type(self._config[key]))
            if isinstance(self._config[key], float):
                # Int can be transformed to float
                type_correct = type_correct or isinstance(value, int)
            if key in self._types:
                type_correct = type_correct or (type(value) in self._types[key])
            assert type_correct, "TypeError: {}:{}".format(key, value)
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

    def register_type(self, key, *types):
        """
        Register special types for item in config. This is used for mixed type declaration.
        """
        assert key in self._config
        self._types[key] = set(types)

    def get_dict(self):
        return self._config

    def extend_config_with_unknown_keys(self, extra_config: dict) -> None:
        """
        This method will merge a new dict to PgConfig without checking the type and existence
        :param extra_config: extra configs
        :return: None
        """
        self._config.update(extra_config)

    def items(self):
        return self._config.items()

    def values(self):
        return self._config.values()

    def keys(self):
        return self._config.keys()

    def pop(self, key):
        self._config.pop(key)
