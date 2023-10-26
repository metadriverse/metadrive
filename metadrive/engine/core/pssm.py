
class PSSMShadow:
    """
    This is the implementation of PSSM for adding shadwo for the scene.
    It is based on https://github.com/el-dee/panda3d-samples
    """
    def __init__(self):
        self.camera_rig = None
        self.split_regions = []

        # Basic PSSM configuration
        self.num_splits = 5
        self.split_resolution = 1024
        self.border_bias = 0.058
        self.fixed_bias = 0.5
        self.use_pssm = True
        self.freeze_pssm = False
        self.fog = True
        self.last_cache_reset = globalClock.get_frame_time()

        # Increase camera FOV as well as the far plane
        self.camLens.set_fov(90)
        self.camLens.set_near_far(0.1, 50000)
