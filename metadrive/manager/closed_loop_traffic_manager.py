from metadrive.manager.scenario_traffic_manager import ScenarioTrafficManager
from metadrive.type import MetaDriveType
from metadrive.component.static_object.traffic_object import TrafficCone, TrafficBarrier
from metadrive.policy.replay_policy import ReplayTrafficParticipantPolicy
from metadrive.engine.logger import get_logger
from metadrive.policy.closed_loop_waypoint_policy import ClosedLoopPolicy
logger = get_logger()


class ClosedLoopTrafficManager(ScenarioTrafficManager):
    """
    Closed loop traffic manager for closed-loop simulation
    Note that this manager is only controlling non-ego agents
    """

    def __init__(self, policy_cls=ClosedLoopPolicy):
        super(ClosedLoopTrafficManager, self).__init__(policy_cls)

    def get_scenario_id_to_obj_id(self):
        return self._scenario_id_to_obj_id
    

    def get_obj_id_to_scenario_id(self):
        return self._obj_id_to_scenario_id


    def after_step(self, *args, **kwargs):
        if self.episode_step < self.current_scenario_length:
            replay_done = False
            for scenario_id, track in self.current_traffic_data.items():
                if scenario_id == self.sdc_scenario_id:
                    continue
                if scenario_id != self.sdc_scenario_id and scenario_id not in self._scenario_id_to_obj_id:
                    if track["type"] == MetaDriveType.VEHICLE:
                        self.spawn_vehicle(scenario_id, track)
                    elif track["type"] == MetaDriveType.CYCLIST:
                        self.spawn_cyclist(scenario_id, track)
                    elif track["type"] == MetaDriveType.PEDESTRIAN:
                        self.spawn_pedestrian(scenario_id, track)
                    elif track["type"] in [MetaDriveType.TRAFFIC_CONE, MetaDriveType.TRAFFIC_BARRIER]:
                        cls = TrafficBarrier if track["type"] == MetaDriveType.TRAFFIC_BARRIER else TrafficCone
                        self.spawn_static_object(cls, scenario_id, track)
                    else:
                        logger.info("Do not support {}".format(track["type"]))

                elif self.has_policy(self._scenario_id_to_obj_id[scenario_id], ClosedLoopPolicy):
                    # static object will not be cleaned!
                    agent_id = self._scenario_id_to_obj_id[scenario_id]
                    policy = self.get_policy(agent_id)
                    if policy.is_current_step_valid:
                        policy.act()
                    else:
                        self._obj_to_clean_this_frame.append(scenario_id)

                else:
                    import pdb; pdb.set_trace()
                    raise ValueError(
                        "The scenario id {} is not in the current traffic data!".format(scenario_id)
                    )

        else:
            replay_done = True
            # clean replay vehicle
            for scenario_id, obj_id in self._scenario_id_to_obj_id.items():
                if self.has_policy(obj_id, ClosedLoopPolicy) and not self.is_static_object(obj_id):
                    self._obj_to_clean_this_frame.append(scenario_id)

        for scenario_id in list(self._obj_to_clean_this_frame):
            obj_id = self._scenario_id_to_obj_id.pop(scenario_id)
            _scenario_id = self._obj_id_to_scenario_id.pop(obj_id)
            assert _scenario_id == scenario_id
            self.clear_objects([obj_id])

        return dict(default_agent=dict(replay_done=replay_done))

    