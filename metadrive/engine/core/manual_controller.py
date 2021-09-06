import logging

from direct.controls.InputState import InputState

from metadrive.utils import is_win, is_mac

if (not is_win()) and (not is_mac()):
    try:
        import evdev
        from evdev import ecodes, InputDevice
    except ImportError:
        pass

from metadrive.utils import import_pygame

pygame = import_pygame()


class Controller:
    def process_input(self, vehicle):
        raise NotImplementedError


class KeyboardController(Controller):
    INCREMENT = 2e-1

    def __init__(self):
        # Input
        # self.pygame_control = True if engine.highway_render is not None else False
        self.pygame_control = False
        if not self.pygame_control:
            self.inputs = InputState()
            self.inputs.watchWithModifiers('forward', 'w')
            self.inputs.watchWithModifiers('reverse', 's')
            self.inputs.watchWithModifiers('turnLeft', 'a')
            self.inputs.watchWithModifiers('turnRight', 'd')

    def process_input(self, vehicle):
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


class SteeringWheelController(Controller):
    RIGHT_SHIFT_PADDLE = 4
    LEFT_SHIFT_PADDLE = 5
    STEERING_MAKEUP = 1.5

    def __init__(self):
        try:
            import evdev
            from evdev import ecodes, InputDevice
        except ImportError:
            print(
                "Fail to load evdev, which is required for steering wheel control. "
                "Install evdev via pip install evdev"
            )
        pygame.display.init()
        pygame.joystick.init()
        assert not is_win(), "Joystick is supported in linux and mac only"
        assert pygame.joystick.get_count() > 0, "Please connect joystick or use keyboard input"
        print("Successfully Connect your Joystick!")

        ffb_device = evdev.list_devices()[0]
        self.ffb_dev = InputDevice(ffb_device)

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        self.right_shift_paddle = False
        self.left_shift_paddle = False

    def process_input(self, vehicle):
        pygame.event.pump()
        steering = -self.joystick.get_axis(0)
        throttle_brake = -self.joystick.get_axis(2) + self.joystick.get_axis(3)
        offset = 30
        val = int(65535 * (vehicle.speed + offset) / (120 + offset)) if vehicle is not None else 0
        self.ffb_dev.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, val)
        self.right_shift_paddle = True if self.joystick.get_button(self.RIGHT_SHIFT_PADDLE) else False
        self.left_shift_paddle = True if self.joystick.get_button(self.LEFT_SHIFT_PADDLE) else False

        return [steering * self.STEERING_MAKEUP, throttle_brake / 2]
