from pgdrive.component.base_class.base_runable import BaseRunnable


class BasePolicy(BaseRunnable):
    def __init__(self, name=None, random_seed=None, config=None):
        BaseRunnable.__init__(self, name, random_seed, config)

    def destroy(self):
        pass

    def reset(self):
        pass

    def before_step(self, *args, **kwargs):
        pass

    def after_step(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass
