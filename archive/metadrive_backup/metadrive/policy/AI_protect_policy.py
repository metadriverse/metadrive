from metadrive.policy.manual_control_policy import ManualControlPolicy


class AIProtectPolicy(ManualControlPolicy):
    """
    This policy can protect Manual control and EnvInputControl
    """
    def act(self, agent_id):
        action = super(AIProtectPolicy, self).act(agent_id)
        vehicle = self.engine.agent_manager.active_agents[agent_id]
        steering = action[0]
        throttle = action[1]
        if not vehicle.expert_takeover:
            # saver can be used for human or another AI
            save_level = self.engine.global_config["save_level"] if not vehicle.expert_takeover else 1.0
            from metadrive.examples.ppo_expert import expert
            try:
                saver_a, obs = expert(vehicle, deterministic=False, need_obs=True)
                obs = obs[0]
            except ValueError:
                print("Expert can not takeover, due to observation space mismathing!")
                saver_a = action
            else:
                if save_level > 0.9:
                    steering = saver_a[0]
                    throttle = saver_a[1]
                elif save_level > 1e-3:
                    heading_diff = vehicle.heading_diff(vehicle.lane) - 0.5
                    f = min(1 + abs(heading_diff) * vehicle.speed_km_h * vehicle.max_speed_km_h, save_level * 10)
                    # for out of road
                    if (obs[0] < 0.04 * f and heading_diff < 0) or (obs[1] < 0.04 * f and heading_diff > 0) or obs[
                        0] <= 1e-3 or \
                            obs[
                                1] <= 1e-3:
                        steering = saver_a[0]
                        throttle = saver_a[1]
                        if vehicle.speed_km_h < 5:
                            throttle = 0.5
                    # for collision
                    lidar_p = self.engine.agent_manager.observations[vehicle.id].cloud_points
                    left = int(vehicle.config["lidar"]["num_lasers"] / 4)
                    right = int(vehicle.config["lidar"]["num_lasers"] / 4 * 3)
                    if min(lidar_p[left - 4:left + 6]) < (save_level + 0.1) / 10 or min(lidar_p[right - 4:right + 6]
                                                                                        ) < (save_level + 0.1) / 10:
                        # lateral safe distance 2.0m
                        steering = saver_a[0]
                    if action[1] >= 0 and saver_a[1] <= 0 and min(min(lidar_p[0:10]), min(lidar_p[-10:])) < save_level:
                        # longitude safe distance 15 m
                        throttle = saver_a[1]

        # indicate if current frame is takeover step
        pre_save = vehicle.takeover
        vehicle.takeover = True if action[0] != steering or action[1] != throttle else False
        self.action_info = {
            "takeover_start": True if not pre_save and vehicle.takeover else False,
            "takeover_end": True if pre_save and not vehicle.takeover else False,
            "takeover": vehicle.takeover if pre_save else False
        }
        action = (steering, throttle) if self.action_info["takeover"] else action
        self.action_info["action"] = action
        return action
