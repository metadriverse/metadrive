from metadrive.base_class.randomizable import Randomizable
from typing import Callable


class Event:
    """
    Event is happed in episode level, all events will be cleared at the start of the episode
    a event/condition func should take its manager instance and other args as input
    """

    def __init__(self, name, condition_func, event_func, cond_args=None, event_args=None):
        self.name = name
        self.event_func = event_func
        self.event_args = event_args or {}
        self.condition_func = condition_func
        self.cond_args = cond_args or {}
        self.triggered = False

    def trigger(self, manager):
        assert isinstance(manager, BaseManager)
        assert not self.triggered, "evnt can not be trigged twiced!"
        self.event_func(manager, **self.event_args)
        self.triggered = True

    def should_be_triggered(self, manager):
        assert isinstance(manager, BaseManager)
        return True if self.condition_func(manager, **self.cond_args) else False


class BaseManager(Randomizable):
    """
    Managers should be created and registered after launching BaseEngine
    """
    PRIORITY = 10  # the engine will call managers according to the priority

    def __init__(self):
        from metadrive.engine.engine_utils import get_engine, engine_initialized
        assert engine_initialized(), "You should not create manager before the initialization of BaseEngine"
        self.engine = get_engine()
        Randomizable.__init__(self, self.engine.global_random_seed)
        self.spawned_objects = {}
        self.events = {}

    @property
    def episode_step(self):
        return self.engine.episode_step

    def _check_and_trigger_events(self):
        """
        This should usually be called in before_step()
        """
        for event in events.items():
            if event.should_be_triggered(self):
                event.trigger(self)

    def get_untriggered_events(self):
        ret = []
        for name, event in self.events.items():
            if not event.trigger:
                ret.append(name)
        return ret

    def get_triggered_events(self):
        ret = []
        for name, event in self.events.items():
            if event.trigger:
                ret.append(name)
        return ret

    def _add_event(self, event):
        assert isinstance(event, Event), "Only event type can be added"
        self.events[event.name] = event

    def _delete_event(self, event_name):
        assert event_name in self.events, "No event in this manager {}".format(self)
        self.events.pop(event_name)

    def before_step(self, *args, **kwargs) -> dict:
        """
        Usually used to set actions for all elements with their policies
        """
        return dict()

    def step(self, *args, **kwargs):
        pass

    def after_step(self, *args, **kwargs) -> dict:
        """
        Update state for this manager after system advancing dt
        """
        return dict()

    def before_reset(self):
        """
        Update episode level config to this manager and clean element or detach element
        """
        self.clear_objects([object_id for object_id in self.spawned_objects.keys()])
        self.spawned_objects = {}

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
        self.engine = None
        self.spawned_objects = None

    def spawn_object(self, object_class, **kwargs):
        """
        Spawn one objects
        """
        object = self.engine.spawn_object(object_class, **kwargs)
        self.spawned_objects[object.id] = object
        return object

    def clear_objects(self, *args, **kwargs):
        """
        Same as the function in engine, clear objects, Return exclude object ids
        """
        exclude_objects = self.engine.clear_objects(*args, **kwargs)
        for obj in exclude_objects:
            self.spawned_objects.pop(obj)
        return exclude_objects

    def get_objects(self, *args, **kwargs):
        return self.engine.get_objects(*args, *kwargs)

    def change_object_name(self, obj, new_name):
        """
        Change the name of one object, Note: it may bring some bugs if abusing!
        """
        self.engine.change_object_name(obj, new_name)
        obj = self.spawned_objects.pop(obj.name)
        self.spawned_objects[new_name] = obj
        obj.name = new_name

    def add_policy(self, object_id, policy):
        self.engine.add_policy(object_id, policy)
