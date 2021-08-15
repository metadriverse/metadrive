from pgdrive.base_class.randomizable import Randomizable
from pgdrive.base_class.configurable import Configurable
from pgdrive.engine.engine_utils import get_engine


class BasePolicy(Randomizable, Configurable):
    def __init__(self, control_object, random_seed=None, config=None):
        Randomizable.__init__(self, random_seed)
        Configurable.__init__(self, config)
        self.engine = get_engine()
        self.control_object = control_object

    def act(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def destroy(self):
        super(BasePolicy, self).destroy()
        self.control_object = None
        self.engine = None
