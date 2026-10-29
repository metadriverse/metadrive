from collections import OrderedDict


class Event:
    """
    Happening in episode level Event is a process for manipulating existing objects.
    All events will be cleared at the start of the episode
    A event/condition func should take its manager instance and other args as input
    """
    def __init__(
        self, name, condition_func, event_func, cond_args=None, event_args=None, trigger_once=True, priority=100
    ):
        assert DeprecationWarning("Not tested yet")
        self.name = name
        self.event_func = event_func
        self.event_args = event_args or {}
        self.condition_func = condition_func
        self.cond_args = cond_args or {}
        self.has_been_triggered = False
        self.trigger_once = trigger_once
        self.priority = priority

    def trigger(self):
        assert not self.has_been_triggered, "evnt can not be trigged twiced when trigger_once == True!"
        self.event_func(**self.event_args)
        self.has_been_triggered = True if self.trigger_once else False

    def should_be_triggered(self):
        return True if self.condition_func(**self.cond_args) else False


class EventManager:
    def __init__(self):
        assert DeprecationWarning("Not tested yet")
        self.events = OrderedDict()

        def _check_and_trigger_events(self):
            """
            This should usually be called in before_step()
            """
            for event in events.items():
                if event.should_be_triggered():
                    event.trigger()

        def get_untriggered_events(self):
            ret = []
            for name, event in self.events.items():
                if not event.has_been_triggered:
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
            self.events = OrderedDict(sorted(self.events.items(), key=lambda key, value: value.priority))

        def _delete_event(self, event_name):
            assert event_name in self.events, "No event in this manager {}".format(self)
            self.events.pop(event_name)
