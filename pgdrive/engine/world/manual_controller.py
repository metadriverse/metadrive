from direct.controls.InputState import InputState

from pgdrive.utils import import_pygame
from pgdrive.engine.world.pg_world import PGWorld

pygame = import_pygame()


class Controller:
    def process_input(self):
        raise NotImplementedError


class KeyboardController(Controller):
    INCREMENT = 2e-1

    def __init__(self):
        # Input
        # self.pygame_control = True if pg_world.highway_render is not None else False
        self.pygame_control = False
        if not self.pygame_control:
            self.inputs = InputState()
            self.inputs.watchWithModifiers('forward', 'w')
            self.inputs.watchWithModifiers('reverse', 's')
            self.inputs.watchWithModifiers('turnLeft', 'a')
            self.inputs.watchWithModifiers('turnRight', 'd')

    def process_input(self):
        if not self.pygame_control:
            steering = 0.0
            throttle_brake = 0.0
            if not self.inputs.isSet('turnLeft') and not self.inputs.isSet('turnRight'):
                steering = 0.0
            else:
                if self.inputs.isSet('turnLeft'):
                    steering = 1.0
                if self.inputs.isSet('turnRight'):
                    steering = -1.0
            if not self.inputs.isSet('forward') and not self.inputs.isSet("reverse"):
                throttle_brake = 0.0
            else:
                if self.inputs.isSet('forward'):
                    throttle_brake = 1.0
                if self.inputs.isSet('reverse'):
                    throttle_brake = -1.0
        else:
            steering = 0.0
            throttle_brake = 0.0
            key_press = pygame.key.get_pressed()
            throttle_brake += key_press[pygame.K_w] - key_press[pygame.K_s]
            steering += key_press[pygame.K_a] - key_press[pygame.K_d]

        return [steering, throttle_brake]


class JoystickController(Controller):
    def __init__(self):
        pygame.display.init()
        pygame.joystick.init()
        assert pygame.joystick.get_count() > 0, "Please connect joystick or use keyboard input"
        print("Successfully Connect your Joystick!")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

    def process_input(self):
        pygame.event.pump()
        steering = -self.joystick.get_axis(0)
        if abs(steering) < 0.1:
            steering = 0
        throttle_brake = -self.joystick.get_axis(4) * abs(self.joystick.get_axis(4))
        return [steering, throttle_brake]
