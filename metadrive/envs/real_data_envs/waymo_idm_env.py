from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.manager.waymo_idm_traffic_manager import WaymoIDMTrafficManager
from metadrive.manager.waymo_traffic_manager import WaymoTrafficManager


class WaymoIDMEnv(WaymoEnv):
    """
    The Traffic in this environment will be controlled by IDM Policy
    """
    def setup_engine(self):
        super(WaymoIDMEnv, self).setup_engine()
        assert not self.config["no_traffic"], "Please set no_traffic to False to use this environment"
        if not self.config['replay']:
            self.engine.update_manager("traffic_manager", WaymoIDMTrafficManager())
        else:
            self.engine.update_manager("traffic_manager", WaymoTrafficManager())
        # self.engine.accept("n", self.next_seed_reset)
        # self.engine.accept("b", self.last_seed_reset)

    # @property
    def next_seed_reset(self):
        # return (self.current_seed + 1) if self.current_seed is not None else 0
        self.reset(self.current_seed + 1)

    # @property
    def last_seed_reset(self):
        # return (self.current_seed - 1) if self.current_seed is not None else 0
        self.reset(self.current_seed - 1)


if __name__ == "__main__":
    asset_path = AssetLoader.asset_path
    env = WaymoIDMEnv(
        {
            "use_render": True,
            # "agent_policy": WaymoIDMPolicy,
            "manual_control": True,
            # "debug":True,
            "no_traffic": False,
            "replay": False,
            "start_case_index": 0,
            "waymo_data_directory": "E:\\PAMI_waymo_data\\coRL_data\\1385_training",
            "case_num": 1000,
            "horizon": 1000,
            # "vehicle_config": dict(show_lidar=True,
            #                        show_lane_line_detector=True,
            #                        show_side_detector=True)F
        }
    )
    success = []
    for i in range(env.config["case_num"]):
        env.reset()
        while True:
            o, r, d, info = env.step([0, 0])
            assert env.observation_space.contains(o)
            c_lane = env.vehicle.lane
            long, lat, = c_lane.local_coordinates(env.vehicle.position)
            if env.config["use_render"]:
                env.render(
                    text={
                        # "routing_lane_idx": env.engine._object_policies[env.vehicle.id].routing_target_lane.index,
                        # "lane_index": env.vehicle.lane_index,
                        # "current_ckpt_index": env.vehicle.navigation.current_checkpoint_lane_index,
                        # "next_ckpt_index": env.vehicle.navigation.next_checkpoint_lane_index,
                        # "ckpts": env.vehicle.navigation.checkpoints,
                        # "lane_heading": c_lane.heading_theta_at(long),
                        # "long": long,
                        # "lat": lat,ﬁ
                        # "v_heading": env.vehicle.heading_theta,
                        "seed": env.engine.global_seed + env.config["start_case_index"],
                        "reward": r,
                    },
                    # mode="top_down",
                    # film_size=(5000, 5000)
                )

                # pygame.image.save(env._top_down_renderer._background_canvas, "render.png")
                # break
            if d:
                if info["arrive_dest"]:
                    print("seed:{}, success".format(env.engine.global_random_seed))
                env.reset()
