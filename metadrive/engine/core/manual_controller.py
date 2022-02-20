import numpy as np
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

    def process_others(self, *args, **kwargs):
        pass


class KeyboardController(Controller):
    STEERING_INCREMENT = 0.04
    STEERING_DECAY = 0.25

    THROTTLE_INCREMENT = 0.1
    THROTTLE_DECAY = 0.2

    BRAKE_INCREMENT = 0.5
    BRAKE_DECAY = 0.5

    def __init__(self, pygame_control):
        self.pygame_control = pygame_control
        if self.pygame_control:
            pygame.init()
        else:
            self.inputs = InputState()
            self.inputs.watchWithModifiers('forward', 'w')
            self.inputs.watchWithModifiers('reverse', 's')
            self.inputs.watchWithModifiers('turnLeft', 'a')
            self.inputs.watchWithModifiers('turnRight', 'd')
        self.steering = 0.
        self.throttle_brake = 0.
        self.np_random = np.random.RandomState(None)

    def process_input(self, vehicle):
        if not self.pygame_control:
            steering = 0.
            throttle_brake = 0.
            if not self.inputs.isSet('turnLeft') and not self.inputs.isSet('turnRight'):
                steering = 0.
            else:
                if self.inputs.isSet('turnLeft'):
                    steering = 1.0
                if self.inputs.isSet('turnRight'):
                    steering = -1.0
            if not self.inputs.isSet('forward') and not self.inputs.isSet("reverse"):
                throttle_brake = 0.
            else:
                if self.inputs.isSet('forward'):
                    throttle_brake = 1.0
                if self.inputs.isSet('reverse'):
                    throttle_brake = -1.0
        else:
            steering = 0.
            throttle_brake = 0.
            key_press = pygame.key.get_pressed()
            throttle_brake += key_press[pygame.K_w] - key_press[pygame.K_s]
            steering += key_press[pygame.K_a] - key_press[pygame.K_d]

        self.further_process(steering, throttle_brake)

        return np.array([self.steering, self.throttle_brake], dtype=np.float64)

    def further_process(self, steering, throttle_brake):
        if steering == 0.:
            if self.steering > 0.:
                self.steering -= self.STEERING_DECAY
                self.steering = max(0., self.steering)
            elif self.steering < 0.:
                self.steering += self.STEERING_DECAY
                self.steering = min(0., self.steering)
        if throttle_brake == 0.:
            if self.throttle_brake > 0.:
                self.throttle_brake -= self.THROTTLE_DECAY
                self.throttle_brake = max(self.throttle_brake, 0.)
            elif self.throttle_brake < 0.:
                self.throttle_brake += self.BRAKE_DECAY
                self.throttle_brake = min(0., self.throttle_brake)

        if steering > 0.:
            self.steering += self.STEERING_INCREMENT if self.steering > 0. else self.STEERING_DECAY
        elif steering < 0.:
            self.steering -= self.STEERING_INCREMENT if self.steering < 0. else self.STEERING_DECAY

        if throttle_brake > 0.:
            self.throttle_brake = max(self.throttle_brake, 0.)
            self.throttle_brake += self.THROTTLE_INCREMENT
        elif throttle_brake < 0.:
            self.throttle_brake = min(self.throttle_brake, 0.)
            self.throttle_brake -= self.BRAKE_INCREMENT

        rand = self.np_random.rand(2, 1) / 10000
        # self.throttle_brake += rand[0]
        self.steering += rand[1]

        self.throttle_brake = min(max(-1., self.throttle_brake), 1.)
        self.steering = min(max(-1., self.steering), 1.)

    def process_others(self, takeover_callback=None):
        """This function allows the outer loop to call callback if some signal is received by the controller."""
        if (takeover_callback is None) or (not self.pygame_control) or (not pygame.get_init()):
            return
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                # Here we allow user to press T for takeover callback.
                takeover_callback()


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

        self.button_circle = False
        self.button_rectangle = False
        self.button_triangle = False
        self.button_x = False

        self.button_up = False
        self.button_down = False
        self.button_right = False
        self.button_left = False

    def process_input(self, vehicle):
        pygame.event.pump()
        steering = -self.joystick.get_axis(0)
        throttle_brake = -self.joystick.get_axis(2) + self.joystick.get_axis(3)
        offset = 30
        val = int(65535 * (vehicle.speed + offset) / (120 + offset)) if vehicle is not None else 0
        self.ffb_dev.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, val)
        self.right_shift_paddle = True if self.joystick.get_button(self.RIGHT_SHIFT_PADDLE) else False
        self.left_shift_paddle = True if self.joystick.get_button(self.LEFT_SHIFT_PADDLE) else False

        self.left_shift_paddle = True if self.joystick.get_button(self.LEFT_SHIFT_PADDLE) else False
        self.left_shift_paddle = True if self.joystick.get_button(self.LEFT_SHIFT_PADDLE) else False

        self.button_circle = True if self.joystick.get_button(2) else False
        self.button_rectangle = True if self.joystick.get_button(1) else False
        self.button_triangle = True if self.joystick.get_button(3) else False
        self.button_x = True if self.joystick.get_button(0) else False

        hat = self.joystick.get_hat(0)
        self.button_up = True if hat[-1] == 1 else False
        self.button_down = True if hat[-1] == -1 else False
        self.button_left = True if hat[0] == -1 else False
        self.button_right = True if hat[0] == 1 else False

        return [steering * self.STEERING_MAKEUP, throttle_brake / 2]
