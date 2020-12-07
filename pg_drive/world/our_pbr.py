# in order to use pbr in opengles pipe on clusters, we temporally inherit from simple pbr

from simplepbr import Pipeline


class OurPipeline(Pipeline):
    def _setup_tonemapping(self):
        # this func cause error under opengles model
        pass
