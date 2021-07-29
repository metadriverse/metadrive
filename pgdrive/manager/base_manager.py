from typing import Callable, Optional
from pgdrive.utils.random import RandomEngine


class BaseManager(RandomEngine):
    """
    Managers should be created and registered after launching BaseEngine
    """
    def __init__(self):
        from pgdrive.utils.engine_utils import get_engine, engine_initialized
        assert engine_initialized(), "You should not create manager before the initialization of BaseEngine"
        self.engine = get_engine()
        super(BaseManager, self).__init__(self.engine.global_random_seed)
        self._spawned_objects = dict()

    def spawn_object(self, object_class, *args, **kwargs):
        """
        Call this func to spawn one object
        :param object_class: object class
        :param kwargs: class init parameters
        :return: object spawned
        """
        if "random_seed" not in kwargs:
            kwargs["random_seed"] = self.randint()
        obj = object_class(*args, **kwargs)
        self._spawned_objects[obj.id] = obj
        return obj

    def get_objects(self, filter_func: Optional[Callable] = None):
        """
        Return objects spawned and managed by this manager, default all objects
        Since we don't expect a iterator, and the number of objects is not so large, we don't use built-in filter()
        :param filter_func: a filter function, only return objects satisfying this condition
        :return: return all objects or objects satisfying the filter_func
        """
        res = dict()
        for id, obj in self._spawned_objects.items():
            if filter_func is None or filter_func(obj):
                res[id] = obj
        return res

    def clear_objects(self, filter_func: Optional[Callable] = None):
        """
        Destroy all self-generated objects or objects satisfying the filter condition
        Since we don't expect a iterator, and the number of objects is not so large, we don't use built-in filter()
        """
        exclude = []
        for id, obj in self._spawned_objects.items():
            if filter_func is None or filter_func(obj):
                obj.destroy()
            exclude.append(id)
        for id in exclude:
            self._spawned_objects.pop(id)

    def before_step(self, *args, **kwargs) -> dict:
        """
        Usually used to set actions for all elements with their policies
        """
        pass

    def step(self, *args, **kwargs):
        """
        TODO Remove in the future?
        """
        pass

    def after_step(self, *args, **kwargs):
        """
        Update state for this manager after system advancing dt
        """
        pass

    def before_reset(self):
        """
        Update episode level config to this manager and clean element or detach element
        """
        pass

    def reset(self):
        """
        Generate objects according to some pre-defined rules
        """
        pass

    def after_reset(self):
        """
        Usually used to record information after all managers called reset(),
        Since reset() of managers may influence each other
        """
        pass

    def destroy(self):
        """
        Destroy manager
        """
        self.clear_objects()
        self._spawned_objects = None
        self.engine = None
