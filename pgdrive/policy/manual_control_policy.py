from pgdrive.engine.core.manual_controller import KeyboardController, JoystickController
from pgdrive.policy.env_input_policy import EnvInputPolicy
from pgdrive.engine.engine_utils import get_global_config


class ManualControlPolicy(EnvInputPolicy):
    def __init__(self):
        super(ManualControlPolicy, self).__init__()
        config = get_global_config()
        if config["manual_control"] and config["use_render"]:
            if config["controller"] == "keyboard":
                self.controller = KeyboardController()
            elif config["controller"] == "joystick":
                self.controller = JoystickController()
            else:
                raise ValueError("No such a controller type: {}".format(self.config["controller"]))

    def act(self, agent_id):
        if self.engine.global_config["manual_control"] and self.engine.agent_manager.get_agent(
                agent_id) is self.engine.current_track_vehicle and not self.engine.main_camera.is_bird_view_camera():
            return self.controller.process_input(self.engine.current_track_vehicle)
        else:
            return super(ManualControlPolicy, self).act(agent_id)
