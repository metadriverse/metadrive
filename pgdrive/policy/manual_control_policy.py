from pgdrive.engine.core.manual_controller import KeyboardController, JoystickController
from pgdrive.policy.base_policy import BasePolicy
from pgdrive.engine.engine_utils import get_global_config


class ManualControlPolicy(BasePolicy):
    def __init__(self):
        super(ManualControlPolicy, self).__init__(control_object=None)
        config = get_global_config()
        if config["manual_control"] and config["use_render"]:
            if config["controller"] == "keyboard":
                self.controller = KeyboardController()
            elif config["controller"] == "joystick":
                self.controller = JoystickController()
            else:
                raise ValueError("No such a controller type: {}".format(self.config["controller"]))

    def act(self, agent_id):
        if self.engine.agent_manager.get_agent(
                agent_id) is self.engine.current_track_vehicle and not self.engine.main_camera.is_bird_view_camera():
            return self.controller.process_input(self.engine.current_track_vehicle)
        else:
            return self.engine.external_actions[agent_id]
