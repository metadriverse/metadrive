from metadrive.engine.core.manual_controller import KeyboardController, SteeringWheelController
from metadrive.utils import clip
from metadrive.examples import expert
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.engine.engine_utils import get_global_config


class ManualControlPolicy(EnvInputPolicy):
    """
    Control the current track vehicle
    """
    def __init__(self):
        super(ManualControlPolicy, self).__init__()
        config = get_global_config()
        self.engine.accept("t", self.toggle_takeover)
        if config["manual_control"] and config["use_render"]:
            if config["controller"] == "keyboard":
                self.controller = KeyboardController()
            elif config["controller"] == "joystick":
                try:
                    self.controller = SteeringWheelController()
                except:
                    print("Load Joystick Error! Fall back to keyboard control")
                    self.controller = KeyboardController()
            else:
                raise ValueError("No such a controller type: {}".format(self.config["controller"]))

    def act(self, agent_id):
        try:
            if self.engine.current_track_vehicle.expert_takeover:
                return expert(self.engine.current_track_vehicle)
        except ValueError:
            # if observation doesn't match, fall back to manual control
            pass
        if self.engine.global_config["manual_control"] and self.engine.agent_manager.get_agent(
                agent_id) is self.engine.current_track_vehicle and not self.engine.main_camera.is_bird_view_camera():
            return self.controller.process_input(self.engine.current_track_vehicle)
        else:
            return super(ManualControlPolicy, self).act(agent_id)

    def toggle_takeover(self):
        if self.engine.current_track_vehicle is not None:
            self.engine.current_track_vehicle.expert_takeover = not self.engine.current_track_vehicle.expert_takeover


class TakeoverPolicy(EnvInputPolicy):
    """
    Record the takeover signal
    """
    def __init__(self):
        super(TakeoverPolicy, self).__init__()
        config = get_global_config()
        if config["manual_control"] and config["use_render"]:
            if config["controller"] == "joystick":
                self.controller = SteeringWheelController()
            else:
                raise ValueError("Takeover policy can only be activated with SteeringWheel")
        self.takeover = False

    def act(self, agent_id):
        agent_action = super(TakeoverPolicy, self).act(agent_id)
        if self.engine.global_config["manual_control"] and self.engine.agent_manager.get_agent(
                agent_id) is self.engine.current_track_vehicle and not self.engine.main_camera.is_bird_view_camera():
            expert_action = self.controller.process_input(self.engine.current_track_vehicle)
            if self.controller.left_shift_paddle or self.controller.right_shift_paddle:
                # if expert_action[0]*agent_action[0]< 0 or expert_action[1]*agent_action[1] < 0:
                self.takeover = True
                return expert_action
        self.takeover = False
        return agent_action
