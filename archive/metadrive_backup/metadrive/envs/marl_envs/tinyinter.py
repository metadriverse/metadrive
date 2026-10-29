import copy

import gymnasium as gym
import numpy as np

from metadrive.envs.marl_envs.marl_intersection import MultiAgentIntersectionEnv
from metadrive.manager.agent_manager import VehicleAgentManager
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.utils import Config
from metadrive.utils.math import Vector, norm, clip


class CommunicationObservation(LidarStateObservation):
    @property
    def observation_space(self):
        shape = list(self.state_obs.observation_space.shape)
        if self.config["vehicle_config"]["lidar"]["num_lasers"] > 0 \
                and self.config["vehicle_config"]["lidar"]["distance"] > 0:
            # Number of lidar rays and distance should be positive!
            lidar_dim = self.config["vehicle_config"]["lidar"]["num_lasers"]
            lidar_dim += self.env.num_agents * (5 + (4 if self.config["lidar"]["add_others_navi"] else 0))
            shape[0] += lidar_dim
        return gym.spaces.Box(-0.0, 1.0, shape=tuple(shape), dtype=np.float32)

    def __init__(self, vehicle_config, env):
        # Restore this to default since we add all others info into obs
        assert vehicle_config["lidar"]["num_others"] == 0
        super(CommunicationObservation, self).__init__(vehicle_config=vehicle_config)
        self.env = env

        self.agent_name_index_mapping = {}
        self.agent_name_slot_mapping = {}

    def observe(self, vehicle):
        if self.env.num_RL_agents != self.env.num_agents:
            agent_name = self.env.agent_manager.object_to_agent(vehicle.name)
            if agent_name not in self.env.agent_manager.RL_agents:
                return None  # Do not compute observation for non RL agents.
        return super(CommunicationObservation, self).observe(vehicle=vehicle)

    def reset(self, env, vehicle=None):
        self.agent_name_index_mapping = {}
        self.agent_name_slot_mapping = {}

    def lidar_observe(self, vehicle):
        other_v_info = []
        lidar_cfg = self.config["vehicle_config"]["lidar"]
        if self.config["vehicle_config"]["lidar"]["num_lasers"] != 0:
            cloud_points, detected_objects = self.engine.get_sensor("lidar").perceive(
                vehicle,
                physics_world=self.engine.physics_world.dynamic_world,
                num_lasers=lidar_cfg["num_lasers"],
                distance=lidar_cfg["distance"],
                show=self.config["vehicle_config"]["show_lidar"]
            )

            other_v_info = self.get_global_info(vehicle)

            other_v_info += self._add_noise_to_cloud_points(
                cloud_points, gaussian_noise=lidar_cfg["gaussian_noise"], dropout_prob=lidar_cfg["dropout_prob"]
            )

            self.cloud_points = cloud_points
            self.detected_objects = detected_objects
        return other_v_info

    def refresh_agent_name_index_mapping(self):
        num_agents = self.env.num_agents
        name_vehicle_mapping = self.env.agent_manager.active_agents
        name_vehicle_mapping = {k: name_vehicle_mapping[k] for k in sorted(name_vehicle_mapping.keys())}

        existing_agents = set(self.agent_name_index_mapping.keys())
        new_agents = set(name_vehicle_mapping.keys())

        if existing_agents == new_agents:
            return name_vehicle_mapping

        old_agent_names = existing_agents.difference(new_agents)

        # Some new vehicles in and probability old vehicle terminates.
        for old_agent_name in old_agent_names:
            # This guy is done

            old_index = self.agent_name_index_mapping.pop(old_agent_name)
            old_slot = self.agent_name_slot_mapping.pop(old_agent_name)

            # Search which new agent is not assigned yet, send old index to it.
            new_agent_names = new_agents.difference(existing_agents)
            for new_agent_name in new_agent_names:
                if new_agent_name not in self.agent_name_index_mapping:
                    # This new guy is not assigned index yet, just give it old index
                    self.agent_name_index_mapping[new_agent_name] = old_index
                    self.agent_name_slot_mapping[new_agent_name] = old_slot
                    new_agents.remove(new_agent_name)
                    break

        # If new agent still exist, assigned new index to it.
        new_agent_names = new_agents.difference(self.agent_name_index_mapping.keys())
        for k in new_agent_names:
            index = int(k.split("agent")[1])
            index = index % num_agents
            self.agent_name_slot_mapping[k] = index

            index = (index + 1) / (num_agents)  # scale to [1/num_agents, 1]. 0 left to no vehicle.
            self.agent_name_index_mapping[k] = index

        # assert len(new_agents.symmetric_difference(self.agent_name_index_mapping.keys())) == 0
        assert all(k in self.agent_name_index_mapping for k in name_vehicle_mapping)
        assert all(k in self.agent_name_slot_mapping.keys() for k in name_vehicle_mapping)
        self.agent_name_index_mapping = {
            k: self.agent_name_index_mapping[k]
            for k in sorted(self.agent_name_index_mapping.keys())
        }
        self.agent_name_slot_mapping = {
            k: self.agent_name_slot_mapping[k]
            for k in sorted(self.agent_name_slot_mapping.keys())
        }
        # print("SLOT:", self.agent_name_slot_mapping)
        # print("AGEN:", self.agent_name_index_mapping)

        return name_vehicle_mapping

    def _process_norm(self, vector, perceive_distance):
        vector_norm = norm(*vector)
        vector = Vector(vector)
        if vector_norm > perceive_distance:
            vector = vector / vector_norm * perceive_distance
        return vector

    def get_global_info(self, ego_vehicle):
        perceive_distance = ego_vehicle.config["lidar"]["distance"]
        # perceive_distance = 20  # m
        speed_scale = 20  # km/h

        name_vehicle_mapping = self.refresh_agent_name_index_mapping()
        add_others_navi = ego_vehicle.config["lidar"]["add_others_navi"]

        # surrounding_vehicles += [None] * num_others
        num_agents = self.env.num_agents

        if add_others_navi:
            res_size = 5 + 4
        else:
            res_size = 5
        res = [0.0] * res_size * num_agents

        for agent_name, vehicle in name_vehicle_mapping.items():
            # for slot_index, agent_name in self.agent_name_slot_mapping.items():
            #     vehicle = name_vehicle_mapping[agent_name]

            slot_index = self.agent_name_slot_mapping[agent_name]

            ego_position = ego_vehicle.position

            assert agent_name in self.agent_name_index_mapping
            res[slot_index * res_size] = self.agent_name_index_mapping[agent_name]

            # assert isinstance(vehicle, IDMVehicle or Base), "Now MetaDrive Doesn't support other vehicle type"

            relative_position = ego_vehicle.convert_to_local_coordinates(vehicle.position, ego_position)
            relative_position = self._process_norm(relative_position, perceive_distance)

            # It is possible that the centroid of other vehicle is too far away from ego but lidar shed on it.
            # So the distance may greater than perceive distance.
            res[slot_index * res_size + 1] = clip((relative_position[0] / perceive_distance + 1) / 2, 0.0, 1.0)
            res[slot_index * res_size + 2] = clip((relative_position[1] / perceive_distance + 1) / 2, 0.0, 1.0)

            relative_velocity = ego_vehicle.convert_to_local_coordinates(vehicle.velocity, ego_vehicle.velocity)
            relative_velocity = self._process_norm(relative_velocity, speed_scale)
            res[slot_index * res_size + 3] = clip((relative_velocity[0] / speed_scale + 1) / 2, 0.0, 1.0)
            res[slot_index * res_size + 4] = clip((relative_velocity[1] / speed_scale + 1) / 2, 0.0, 1.0)

            if add_others_navi:
                ckpt1, ckpt2 = vehicle.navigation.get_checkpoints()

                relative_ckpt1 = ego_vehicle.lidar._project_to_vehicle_system(ckpt1, ego_vehicle, perceive_distance)
                relative_ckpt1 = self._process_norm(relative_ckpt1, perceive_distance)
                res[slot_index * res_size + 5] = clip((relative_ckpt1[0] / perceive_distance + 1) / 2, 0.0, 1.0)
                res[slot_index * res_size + 6] = clip((relative_ckpt1[1] / perceive_distance + 1) / 2, 0.0, 1.0)

                relative_ckpt2 = ego_vehicle.lidar._project_to_vehicle_system(ckpt2, ego_vehicle)
                relative_ckpt2 = self._process_norm(relative_ckpt2, perceive_distance, perceive_distance)
                res[slot_index * res_size + 7] = clip((relative_ckpt2[0] / perceive_distance + 1) / 2, 0.0, 1.0)
                res[slot_index * res_size + 8] = clip((relative_ckpt2[1] / perceive_distance + 1) / 2, 0.0, 1.0)

        # assert len(res) == len(name_vehicle_mapping) * (5 + (4 if add_others_navi else 0))
        return res


class TinyInterRuleBasedPolicy(IDMPolicy):
    """No IDM and PID are used in this Policy!"""
    def __init__(self, control_object, random_seed, target_speed=10):
        super(TinyInterRuleBasedPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.target_speed = target_speed  # Set to 10km/h. Default is 30km/h.

    def act(self, *args, **kwargs):
        self.move_to_next_road()

        target_lane = self.routing_target_lane
        long, lat = target_lane.local_coordinates(self.control_object.position)

        increment = self.target_speed / 3.6 * \
                    self.engine.global_config["physics_world_step_size"] * \
                    self.engine.global_config["decision_repeat"]

        new_long = long + increment
        new_pos = target_lane.position(new_long, lat)

        new_heading = target_lane.heading_theta_at(new_long + 1)
        self.control_object.set_heading_theta(new_heading)

        # Note: the last_position is not correctly set since in vehicle.before_step
        # we will set last_position to position at that time, which is the latest position.
        # This leads to wrong reward for IDM agent.
        # You need to comment out the line in before_step to make it correct!
        self.control_object.last_position = self.control_object.position
        self.control_object.set_position(new_pos)

        return [0, 0]


class MixedIDMAgentManager(VehicleAgentManager):
    """In this manager, we can replace part of RL policy by IDM policy"""
    def __init__(self, init_observations, num_RL_agents, ignore_delay_done=None, target_speed=10):
        super(MixedIDMAgentManager, self).__init__(init_observations=init_observations)
        self.num_RL_agents = num_RL_agents
        self.RL_agents = set()
        self.dying_RL_agents = set()
        self.all_previous_RL_agents = set()
        self.ignore_delay_done = ignore_delay_done
        self.target_speed = target_speed

    def filter_RL_agents(self, source_dict, original_done_dict=None):

        # new_ret = {k: v for k, v in source_dict.items() if k in self.RL_agents}

        new_ret = {k: v for k, v in source_dict.items() if k in self.all_previous_RL_agents}

        # if len(new_ret) > self.num_RL_agents:
        #     assert len(self.RL_agents) - len(self.dying_RL_agents) == self.num_RL_agents
        # if len(new_ret) < self.num_RL_agents:
        #     print("Something wrong!")
        return new_ret

    def get_observation_spaces(self):
        ret = self.filter_RL_agents(super(MixedIDMAgentManager, self).get_observation_spaces())
        if len(ret) == 0:
            k, v = list(super(MixedIDMAgentManager, self).get_observation_spaces().items())[0]
            return {k: v}
        else:
            return ret

    def get_action_spaces(self):
        ret = self.filter_RL_agents(super(MixedIDMAgentManager, self).get_action_spaces())
        if len(ret) == 0:
            k, v = list(super(MixedIDMAgentManager, self).get_action_spaces().items())[0]
            return {k: v}
        else:
            return ret

    def _finish(self, agent_name, ignore_delay_done=False):
        # ignore_delay_done = True
        if self.ignore_delay_done is not None:
            ignore_delay_done = self.ignore_delay_done
        if agent_name in self.RL_agents:
            self.dying_RL_agents.add(agent_name)
        super(MixedIDMAgentManager, self)._finish(agent_name, ignore_delay_done)

    def _remove_vehicle(self, v):
        agent_name = self.object_to_agent(v.name)
        if agent_name in self.RL_agents:
            self.RL_agents.remove(agent_name)
        if agent_name in self.dying_RL_agents:
            self.dying_RL_agents.remove(agent_name)
        super(MixedIDMAgentManager, self)._remove_vehicle(v)

    def _create_agents(self, config_dict: dict):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type, vehicle_type
        ret = {}
        for agent_count, (agent_id, v_config) in enumerate(config_dict.items()):
            if self.engine.global_config["random_agent_model"]:
                v_type = random_vehicle_type(self.np_random)
            else:
                if v_config.get("vehicle_model", False):
                    v_type = vehicle_type[v_config["vehicle_model"]]
                else:
                    v_type = vehicle_type["default"]
            obj = self.spawn_object(v_type, vehicle_config=v_config)
            ret[agent_id] = obj
            if (len(self.RL_agents) - len(self.dying_RL_agents)) >= self.num_RL_agents:
                # policy = IDMPolicy(obj, self.generate_seed())
                policy_cls = TinyInterRuleBasedPolicy
                obj._use_special_color = False
                self.add_policy(obj.id, policy_cls, obj, self.generate_seed(), target_speed=self.target_speed)
            else:
                policy_cls = self.agent_policy
                self.RL_agents.add(agent_id)
                self.all_previous_RL_agents.add(agent_id)
                obj._use_special_color = True
                self.add_policy(obj.id, policy_cls, obj, self.generate_seed())
        return ret

    def reset(self):
        self.RL_agents.clear()
        self.dying_RL_agents.clear()
        self.all_previous_RL_agents.clear()
        super(MixedIDMAgentManager, self).reset()

    @property
    def allow_respawn(self):
        """In native implementation, we can only respawn new vehicle when the total vehicles in the scene
        is less then num_agents. But here, since we always have N-K IDM vehicles and K RL vehicles, so
        allow_respawn is always False. We set to True to allow spawning new vehicle.
        """
        if self._allow_respawn is False:
            return False

        if len(self.active_agents) < self.engine.global_config["num_agents"]:
            return True

        if len(self.RL_agents) - len(self.dying_RL_agents) < self.num_RL_agents:
            return True

        return False


class MultiAgentTinyInter(MultiAgentIntersectionEnv):
    @staticmethod
    def default_config() -> Config:
        tiny_config = dict(
            success_reward=10.0,

            # Default MARL MetaDrive setting:
            out_of_road_penalty=10,
            crash_vehicle_penalty=10,
            crash_object_penalty=10,

            # Default single-agent MetaDrive setting:
            # out_of_road_penalty=5.0,
            # crash_vehicle_penalty=5.0,
            # crash_object_penalty=5.0,
            num_agents=8,
            num_RL_agents=8,
            map_config=dict(
                exit_length=30,
                lane_num=1,
                lane_width=4,

                # Additional config to control the radius
                radius=None
            ),

            # Whether to remove dead vehicles immediately
            ignore_delay_done=True,

            # The target speed of IDM agents, if any
            target_speed=10,

            # Whether to use full global relative information as obs
            use_communication_obs=False,
        )
        return MultiAgentIntersectionEnv.default_config().update(tiny_config, allow_add_new_key=True)

    def _get_reset_return(self, reset_info):
        org = super(MultiAgentTinyInter, self)._get_reset_return(reset_info)

        # if self.num_RL_agents == self.num_agents:
        #     return org

        return self.agent_manager.filter_RL_agents(org[0])

    def step(self, actions):
        o, r, tm, tc, i = super(MultiAgentTinyInter, self).step(actions)

        # if self.num_RL_agents == self.num_agents:
        #     return o, r, tm, tc, i

        original_done_dict = copy.deepcopy(tm)
        d = self.agent_manager.filter_RL_agents(tm, original_done_dict=original_done_dict)
        if "__all__" in d:
            d.pop("__all__")
        # assert len(d) == self.agent_manager.num_RL_agents, d
        d["__all__"] = all(d.values())
        return (
            self.agent_manager.filter_RL_agents(o, original_done_dict=original_done_dict),
            self.agent_manager.filter_RL_agents(r, original_done_dict=original_done_dict),
            d,
            self.agent_manager.filter_RL_agents(tc, original_done_dict=original_done_dict),
            self.agent_manager.filter_RL_agents(i, original_done_dict=original_done_dict),
        )

    def _preprocess_actions(self, actions):
        if self.num_RL_agents == self.num_agents:
            return super(MultiAgentTinyInter, self)._preprocess_actions(actions)

        actions = {v_id: actions[v_id] for v_id in self.agents.keys() if v_id in self.agent_manager.RL_agents}
        return actions

    def __init__(self, config=None):
        super(MultiAgentTinyInter, self).__init__(config=config)
        self.num_RL_agents = self.config["num_RL_agents"]
        # if self.num_RL_agents == self.num_agents:  # Not using mixed traffic and only RL agents are running.
        #     pass
        # else:
        self.agent_manager = MixedIDMAgentManager(
            init_observations=self._get_observations(),
            num_RL_agents=self.num_RL_agents,
            ignore_delay_done=self.config["ignore_delay_done"],
            target_speed=self.config["target_speed"]
        )

    def get_single_observation(self):
        if self.config["use_communication_obs"]:
            return CommunicationObservation(self.config, self)
        else:
            return LidarStateObservation(self.config)


if __name__ == '__main__':
    env = MultiAgentTinyInter(
        config={
            "num_agents": 4,
            "num_RL_agents": 4,
            "use_render": True,

            # "map_config": {"radius": 50},
            # "ignore_delay_done": True,
            # "use_communication_obs": True,
            "vehicle_config": {
                "show_lidar": True,
                "show_line_to_dest": True,
                "lidar": {
                    "num_others": 2,
                    "add_others_navi": True
                }
            },
            # "manual_control": True,
            # "use_render": True,

            # "debug_static_world": True,
        }
    )
    o = env.reset()
    # env.engine.force_fps.toggle()
    print("vehicle num", len(env.engine.traffic_manager.vehicles))
    print("RL agent num", len(o))
    ep_success = 0
    ep_done = 0
    ep_reward_sum = 0.0
    ep_success_reward_sum = 0.0
    for i in range(1, 100000):
        o, r, tm, tc, info = env.step({k: [0.0, 0.0] for k in env.action_space.sample().keys()})
        # env.render("top_down", camera_position=(42.5, 0), film_size=(500, 500))
        vehicles = env.agents

        for k, v in tm.items():
            if v and k in info:
                ep_success += int(info[k]["arrive_dest"])
                ep_reward_sum += int(info[k]["episode_reward"])
                ep_done += 1
                if info[k]["arrive_dest"]:
                    ep_success_reward_sum += int(info[k]["episode_reward"])

        # if not te["__all__"]:
        #     assert sum(
        #         [env.engine.get_policy(v.name).__class__.__name__ == "EnvInputPolicy" for k, v in vehicles.items()]
        #     ) == env.config["num_RL_agents"]
        if any(tm.values()):
            print("Somebody dead.", tm, info)
            # print("Step {}. Policies: {}".format(i, {k: v['policy'] for k, v in info.items()}))
        if tm["__all__"]:
            # assert i >= 1000
            print("Reset. ", i, info)
            # break
            print(
                "Success Rate: {:.3f}, reward: {:.3f}, success reward: {:.3f}, failed reward: {:.3f}, total num {}".
                format(
                    ep_success / ep_done if ep_done > 0 else -1, ep_reward_sum / ep_done if ep_done > 0 else -1,
                    ep_success_reward_sum / ep_success if ep_success > 0 else -1,
                    (ep_reward_sum - ep_success_reward_sum) / (ep_done - ep_success) if
                    (ep_done - ep_success) > 0 else -1, ep_done
                )
            )
            break
            env.reset()
    env.close()
