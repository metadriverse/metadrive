import logging

from pgdrive.envs.pgdrive_env_v2 import PGDriveEnvV2
from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.scene_creator.road.road import Road
from pgdrive.scene_creator.vehicle.base_vehicle import BaseVehicle
from pgdrive.scene_manager.spawn_manager import SpawnManager
from pgdrive.utils import setup_logger, get_np_random, PGConfig
from pgdrive.utils.pg_config import merge_dicts

MULTI_AGENT_PGDRIVE_DEFAULT_CONFIG = dict(
    # ===== Multi-agent =====
    is_multi_agent=True,
    num_agents=2,  # If num_agents is set to None, then endless vehicles will be added only the empty spawn points exist

    # Whether to terminate a vehicle if it crash with others. Since in MA env the crash is extremely dense, so
    # frequently done might not be a good idea.
    crash_done=False,
    out_of_road_done=True,

    # Whether the vehicle can rejoin the episode
    allow_respawn=True,

    # The maximum length of the episode. If allow respawn, then this is the maximum step that respawn can happen. After
    # that, the episode won't terminate until all existing vehicles reach their horizon or done. The vehicle specified
    # horizon is also this value.
    horizon=1000,

    # ===== Vehicle Setting =====
    vehicle_config=dict(lidar=dict(num_lasers=72, distance=40, num_others=0)),
    target_vehicle_configs=dict(),

    # ===== New Reward Setting =====
    out_of_road_penalty=5.0,
    crash_vehicle_penalty=5.0,
    crash_object_penalty=5.0,

    # ===== Environmental Setting =====
    top_down_camera_initial_x=0,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=120,  # height
    traffic_density=0.,
    auto_termination=False,
    camera_height=4,
)


class MultiAgentPGDrive(PGDriveEnvV2):
    """
    This serve as the base class for Multi-agent PGDrive!
    """

    # A list of road instances denoting which roads afford spawn points. If not set, then search for all
    # possible roads and spawn new agents in them if possible.
    spawn_roads = [
        # Road(FirstBlock.NODE_1, FirstBlock.NODE_2),
        Road(FirstBlock.NODE_2, FirstBlock.NODE_3)
    ]

    @staticmethod
    def default_config() -> PGConfig:
        config = PGDriveEnvV2.default_config()
        config.update(MULTI_AGENT_PGDRIVE_DEFAULT_CONFIG)
        return config

    def __init__(self, config=None):
        super(MultiAgentPGDrive, self).__init__(config)
        self._top_down_renderer = None

    def _process_extra_config(self, config) -> "PGConfig":
        ret_config = self.default_config().update(
            config, allow_overwrite=False, stop_recursive_update=["target_vehicle_configs"]
        )
        if not ret_config["crash_done"] and ret_config["crash_vehicle_penalty"] > 2:
            logging.warning(
                "Are you sure you wish to set crash_vehicle_penalty={} when crash_done=False?".format(
                    ret_config["crash_vehicle_penalty"]
                )
            )
        if ret_config["use_render"] and ret_config["fast"]:
            logging.warning("Turn fast=False can accelerate Multi-agent rendering performance!")

        # Workaround
        if ret_config["target_vehicle_configs"]:
            for k, v in ret_config["target_vehicle_configs"].items():
                old = ret_config["vehicle_config"].copy()
                new = old.update(v)
                ret_config["target_vehicle_configs"][k] = new

        self._spawn_manager = SpawnManager(
            exit_length=ret_config["map_config"]["exit_length"],
            lane_num=ret_config["map_config"]["lane_num"],
            num_agents=ret_config["num_agents"],
            vehicle_config=ret_config["vehicle_config"],
            target_vehicle_configs=ret_config["target_vehicle_configs"],
        )

        self._spawn_manager.set_spawn_roads(self.spawn_roads)

        ret_config = self._update_agent_pos_configs(ret_config)
        return ret_config

    def _update_agent_pos_configs(self, config):
        config["target_vehicle_configs"] = self._spawn_manager.get_target_vehicle_configs(seed=self._DEBUG_RANDOM_SEED)
        return config

    def done_function(self, vehicle_id):
        done, done_info = super(MultiAgentPGDrive, self).done_function(vehicle_id)
        if done_info["crash"] and (not self.config["crash_done"]):
            assert done_info["crash_vehicle"] or done_info["arrive_dest"] or done_info["out_of_road"]
            if not (done_info["arrive_dest"] or done_info["out_of_road"]):
                # Does not revert done if high-priority termination happens!
                done = False

        if done_info["out_of_road"] and (not self.config["out_of_road_done"]):
            assert done_info["crash_vehicle"] or done_info["arrive_dest"] or done_info["out_of_road"]
            if not done_info["arrive_dest"]:
                done = False

        return done, done_info

    def step(self, actions):
        o, r, d, i = super(MultiAgentPGDrive, self).step(actions)
        o, r, d, i = self._after_vehicle_done(o, r, d, i)

        # Update respawn manager
        if self.episode_steps >= self.config["horizon"]:
            self._agent_manager.set_allow_respawn(False)
        self._spawn_manager.step()
        new_obs_dict = self._respawn_vehicles()
        if new_obs_dict:
            for new_id, new_obs in new_obs_dict.items():
                o[new_id] = new_obs
                r[new_id] = 0.0
                i[new_id] = {}
                d[new_id] = False

        # Update __all__
        d["__all__"] = (
            ((self.episode_steps >= self.config["horizon"]) and (all(d.values()))) or (len(self.vehicles) == 0)
            or (self.episode_steps >= 5 * self.config["horizon"])
        )
        if d["__all__"]:
            for k in d.keys():
                d[k] = True

        return o, r, d, i

    def reset(self, *args, **kwargs):
        self.config = self._update_agent_pos_configs(self.config)
        ret = super(MultiAgentPGDrive, self).reset(*args, **kwargs)
        assert len(self.vehicles) == self.num_agents
        return ret

    def _reset_agents(self):
        # update config (for new possible spawn places)
        for v_id, v in self.vehicles.items():
            v.vehicle_config = self._get_target_vehicle_config(self.config["target_vehicle_configs"][v_id])
        super(MultiAgentPGDrive, self)._reset_agents()  # Update config before actually resetting!
        self.for_each_vehicle(self._update_destination_for)

    def _after_vehicle_done(self, obs=None, reward=None, dones: dict = None, info=None):
        for v_id, v_info in info.items():
            if v_info.get("episode_length", 0) >= self.config["horizon"]:
                if dones[v_id] is not None:
                    info[v_id]["max_step"] = True
                    dones[v_id] = True
                    self.dones[v_id] = True
        for dead_vehicle_id, done in dones.items():
            if done:
                self._agent_manager.finish(dead_vehicle_id)
                self._update_camera_after_finish(dead_vehicle_id)
        return obs, reward, dones, info

    def _update_camera_after_finish(self, dead_vehicle_id):
        if self.main_camera is not None and dead_vehicle_id == self._agent_manager.object_to_agent(
                self.current_track_vehicle.name) \
                and self.pg_world.taskMgr.hasTaskNamed(self.main_camera.CHASE_TASK_NAME):
            self.chase_another_v()

    def _get_vehicles(self):
        return {
            name: BaseVehicle(self.pg_world, self._get_target_vehicle_config(new_config))
            for name, new_config in self.config["target_vehicle_configs"].items()
        }

    def _get_observations(self):
        return {
            name: self.get_single_observation(self._get_target_vehicle_config(new_config))
            for name, new_config in self.config["target_vehicle_configs"].items()
        }

    def _get_target_vehicle_config(self, extra_config: dict):
        """
        Newly introduce method
        """
        vehicle_config = merge_dicts(self.config["vehicle_config"], extra_config, allow_new_keys=False)
        return PGConfig(vehicle_config)

    def _after_lazy_init(self):
        super(MultiAgentPGDrive, self)._after_lazy_init()

        # Use top-down view by default
        if hasattr(self, "main_camera") and self.main_camera is not None:
            top_down_camera_height = self.config["top_down_camera_initial_z"]
            self.main_camera.camera.setPos(0, 0, top_down_camera_height)
            self.main_camera.top_down_camera_height = top_down_camera_height
            self.main_camera.stop_track(self.pg_world, self.current_track_vehicle)
            self.main_camera.camera_x += self.config["top_down_camera_initial_x"]
            self.main_camera.camera_y += self.config["top_down_camera_initial_y"]

    def _respawn_vehicles(self):
        new_obs_dict = {}
        while True:
            new_id, new_obs = self._respawn_single_vehicle()
            if new_obs is not None:
                new_obs_dict[new_id] = new_obs
            else:
                break
        return new_obs_dict

    def _force_respawn(self, agent_name):
        """
        This function can force a given vehicle to respawn!
        """
        self._agent_manager.finish(agent_name)
        self._update_camera_after_finish(agent_name)
        new_id, new_obs = self._respawn_single_vehicle()
        return new_id, new_obs

    def _respawn_single_vehicle(self):
        """
        Arbitrary insert a new vehicle to a new spawn place if possible.
        """
        safe_places_dict = self._spawn_manager.get_available_spawn_places(self.pg_world, self.current_map)
        if len(safe_places_dict) == 0 or not self._agent_manager.allow_respawn:
            # No more run, just wait!
            return None, None
        assert len(safe_places_dict) > 0
        bp_index = get_np_random(self._DEBUG_RANDOM_SEED).choice(list(safe_places_dict.keys()), 1)[0]
        new_spawn_place = safe_places_dict[bp_index]

        if new_spawn_place[self._spawn_manager.FORCE_AGENT_NAME] is not None:
            if new_spawn_place[self._spawn_manager.FORCE_AGENT_NAME] != self._agent_manager.next_agent_id():
                return None, None

        new_agent_id, vehicle = self._agent_manager.propose_new_vehicle()
        new_spawn_place_config = new_spawn_place["config"]
        vehicle.vehicle_config.update(new_spawn_place_config)
        vehicle.reset(self.current_map)
        self._update_destination_for(vehicle)
        vehicle.update_state(detector_mask=None)
        self.dones[new_agent_id] = False  # Put it in the internal dead-tracking dict.

        new_obs = self.observations[new_agent_id].observe(vehicle)
        return new_agent_id, new_obs

    def _update_destination_for(self, vehicle):
        pass

        # when agent re-joined to the game, call this to set the new route to destination
        # end_road = -get_np_random(self._DEBUG_RANDOM_SEED).choice(self.spawn_roads)  # Use negative road!
        # vehicle.routing_localization.set_route(vehicle.lane_index[0], end_road.end_node)

    def render(self, mode='human', text=None, *args, **kwargs):
        if mode == "top_down":
            ret = self._render_topdown(*args, **kwargs)
        else:
            ret = super(MultiAgentPGDrive, self).render(mode=mode, text=text)
        return ret

    def _render_topdown(self, *args, **kwargs):
        if self._top_down_renderer is None:
            from pgdrive.obs.top_down_renderer import TopDownRenderer
            self._top_down_renderer = TopDownRenderer(self.current_map, *args, **kwargs)
        self._top_down_renderer.render(list(self.vehicles.values()))


def _test():
    setup_logger(True)
    env = MultiAgentPGDrive(
        {
            "num_agents": 12,
            "allow_respawn": False,
            "use_render": True,
            "debug": False,
            "fast": True,
            "manual_control": True,
            "pg_world_config": {
                "pstats": False
            },
        }
    )
    o = env.reset()
    total_r = 0
    for i in range(1, 100000):
        # o, r, d, info = env.step(env.action_space.sample())
        o, r, d, info = env.step({v_id: [0, 1] for v_id in env.vehicles.keys()})
        for r_ in r.values():
            total_r += r_
        # o, r, d, info = env.step([0,1])
        d.update({"total_r": total_r})
        # env.render(text=d)
        if len(env.vehicles) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()


def _vis():
    setup_logger(True)
    env = MultiAgentPGDrive(
        {
            # "use_render": True,
            # "fast": True,
            "num_agents": 12,
            "allow_respawn": False,
            "manual_control": True,
            "pg_world_config": {
                "pstats": False
            },
        }
    )
    o = env.reset()
    total_r = 0
    for i in range(1, 100000):
        # o, r, d, info = env.step(env.action_space.sample())
        o, r, d, info = env.step({v_id: [0.0, 0.0] for v_id in env.vehicles.keys()})
        for r_ in r.values():
            total_r += r_
        # o, r, d, info = env.step([0,1])
        # d.update({"total_r": total_r})
        env.render(mode="top_down")
        if len(env.vehicles) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()


if __name__ == '__main__':
    _vis()
