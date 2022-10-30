from metadrive.engine.core.manual_controller import KeyboardController, SteeringWheelController, XboxController
from metadrive.engine.engine_utils import get_global_config
from metadrive.examples import expert
from metadrive.policy.env_input_policy import EnvInputPolicy


class ManualControlPolicy(EnvInputPolicy):
    """
    Control the current track vehicle
    """
    def __init__(self, obj, seed):
        super(ManualControlPolicy, self).__init__(obj, seed)
        config = self.engine.global_config

        if config["manual_control"] and config["use_render"]:
            self.engine.accept("t", self.toggle_takeover)
            pygame_control = False
        elif config["manual_control"]:
            # Use pygame to accept key strike.
            pygame_control = True
        else:
            pygame_control = False

        # if config["manual_control"] and config["use_render"]:
        if config["manual_control"]:
            if config["controller"] == "keyboard":
                self.controller = KeyboardController(pygame_control=pygame_control)
            elif config["controller"] in ["xboxController", "xboxcontroller", "xbox"]:
                try:
                    if config["controller"] == "joystick":
                        self.controller = SteeringWheelController()
                    else:
                        self.controller = XboxController()
                except:
                    print("Load Joystick Error! Fall back to keyboard control")
                    self.controller = KeyboardController(pygame_control=pygame_control)
            else:
                raise ValueError("No such a controller type: {}".format(self.config["controller"]))
        else:
            self.controller = None

    def act(self, agent_id):

        self.controller.process_others(takeover_callback=self.toggle_takeover)

        try:
            if self.engine.current_track_vehicle.expert_takeover:
                return expert(self.engine.current_track_vehicle)
        except (ValueError, AssertionError):
            # if observation doesn't match, fall back to manual control
            print("Current observation does not match the format that expert can accept.")
            self.toggle_takeover()

        is_track_vehicle = self.engine.agent_manager.get_agent(agent_id) is self.engine.current_track_vehicle
        not_in_native_bev = (self.engine.main_camera is None) or (not self.engine.main_camera.is_bird_view_camera())
        if self.engine.global_config["manual_control"] and is_track_vehicle and not_in_native_bev:
            return self.controller.process_input(self.engine.current_track_vehicle)
        else:
            return super(ManualControlPolicy, self).act(agent_id)

    def toggle_takeover(self):
        if self.engine.current_track_vehicle is not None:
            self.engine.current_track_vehicle.expert_takeover = not self.engine.current_track_vehicle.expert_takeover
            print("The expert takeover is set to: ", self.engine.current_track_vehicle.expert_takeover)


class TakeoverPolicy(EnvInputPolicy):
    """
    Record the takeover signal
    """
    def __init__(self, obj, seed):
        super(TakeoverPolicy, self).__init__(obj, seed)
        config = get_global_config()
        if config["manual_control"] and config["use_render"]:
            if config["controller"] == "joystick":
                self.controller = SteeringWheelController()
            elif config["controller"] == "keyboard":
                self.controller = KeyboardController(False)
            elif config["controller"] == "xboxController":
                self.controller = XboxController()
            else:
                raise ValueError("Unknown Policy: {}".format(config["controller"]))
        self.takeover = False

    def act(self, agent_id):
        agent_action = super(TakeoverPolicy, self).act(agent_id)
        if self.engine.global_config["manual_control"] and self.engine.agent_manager.get_agent(
                agent_id) is self.engine.current_track_vehicle and not self.engine.main_camera.is_bird_view_camera():
            expert_action = self.controller.process_input(self.engine.current_track_vehicle)
            if isinstance(self.controller, SteeringWheelController) and (self.controller.left_shift_paddle
                                                                         or self.controller.right_shift_paddle):
                # if expert_action[0]*agent_action[0]< 0 or expert_action[1]*agent_action[1] < 0:
                self.takeover = True
                return expert_action
            elif isinstance(self.controller, KeyboardController) and abs(sum(expert_action)) > 1e-2:
                self.takeover = True
                return expert_action
            elif isinstance(self.controller, XboxController) and (self.controller.button_x or self.controller.button_y):
                self.takeover = True
                return expert_action
        self.takeover = False
        return agent_action
