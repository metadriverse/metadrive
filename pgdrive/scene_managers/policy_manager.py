from pgdrive.scene_managers.base_manager import BaseManager


class PolicyManager(BaseManager):
    def __init__(self):
        super(PolicyManager, self).__init__()
        self._policy_name_to_vehicle = {}
        self._vehicle_name_to_policy = {}

    def reset(self):
        """
        Some policy might be stateful, for example a LSTM-based neural policy network. We need to reset the states
        of all policies here.
        """
        for p in self._spawned_objects.values():
            p.reset()
        print('222')

    def register_new_policy(self, policy_class, vehicle, traffic_manager, *args, **kwargs):
        # e = get_pgdrive_engine()
        # TODO: We should have a general "vehicle manager". Then we can get the BaseVehicle instance from
        #  engine according to agent_name! Here is only a workaround.
        policy = self.spawn_object(policy_class, vehicle=vehicle, traffic_manager=traffic_manager, *args, **kwargs)
        policy_name = policy.name
        self._policy_name_to_vehicle[policy_name] = vehicle
        self._vehicle_name_to_policy[vehicle.name] = policy

    def get_policy(self, vehicle_name):
        # TODO 2 (pzh): We should have a remove machanism! The vehicle is always
        #   stored in the dict creating potential leak!
        # TODO(pzh) I am not sure yet to use object name or agent name here!
        if vehicle_name not in self._vehicle_name_to_policy:
            return None
        return self._vehicle_name_to_policy[vehicle_name]

    def destroy(self):
        for p in self._spawned_objects.values():
            if hasattr(p, "destroy"):
                p.destroy()
        super(PolicyManager, self).destroy()
