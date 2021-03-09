from typing import Tuple, Union

import numpy as np

from pgdrive.utils import import_pygame

pygame = import_pygame()

PositionType = Union[Tuple[float, float], np.ndarray]


class WorldSurface(pygame.Surface):
    """
    A pygame Surface implementing a local coordinate system so that we can move and zoom in the displayed area.
    From highway-env, See more information on its Github page: https://github.com/eleurent/highway-env.
    """

    BLACK = (0, 0, 0)
    GREY = (100, 100, 100)
    GREEN = (50, 200, 0)
    YELLOW = (200, 200, 0)
    WHITE = (255, 255, 255)
    INITIAL_SCALING = 5.5
    INITIAL_CENTERING = [0.5, 0.5]
    SCALING_FACTOR = 1.3
    MOVING_FACTOR = 0.1

    def __init__(self, size: Tuple[int, int], flags: object, surf: pygame.SurfaceType) -> None:
        surf.fill(pygame.Color("Black"))
        super().__init__(size, flags, surf)
        self.origin = np.array([0, 0])
        self.scaling = self.INITIAL_SCALING
        self.centering_position = self.INITIAL_CENTERING
        self.fill(self.BLACK)

    def pix(self, length: float) -> int:
        """
        Convert a distance [m] to pixels [px].

        :param length: the input distance [m]
        :return: the corresponding size [px]
        """
        return int(length * self.scaling)

    def pos2pix(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert two world coordinates [m] into a position in the surface [px]

        :param x: x world coordinate [m]
        :param y: y world coordinate [m]
        :return: the coordinates of the corresponding pixel [px]
        """
        return self.pix(x - self.origin[0]), self.pix(y - self.origin[1])

    def vec2pix(self, vec: PositionType) -> Tuple[int, int]:
        """
        Convert a world position [m] into a position in the surface [px].

        :param vec: a world position [m]
        :return: the coordinates of the corresponding pixel [px]
        """
        return self.pos2pix(vec[0], vec[1])

    def is_visible(self, vec: PositionType, margin: int = 50) -> bool:
        """
        Is a position visible in the surface?
        :param vec: a position
        :param margin: margins around the frame to test for visibility
        :return: whether the position is visible
        """
        x, y = self.vec2pix(vec)
        return -margin < x < self.get_width() + margin and -margin < y < self.get_height() + margin

    def move_display_window_to(self, position: PositionType) -> None:
        """
        Set the origin of the displayed area to center on a given world position.

        :param position: a world position [m]
        """
        self.origin = position - np.array(
            [
                self.centering_position[0] * self.get_width() / self.scaling,
                self.centering_position[1] * self.get_height() / self.scaling
            ]
        )

    def handle_event(self, event: pygame.event.EventType) -> None:
        """
        Handle pygame events for moving and zooming in the displayed area.

        :param event: a pygame event
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_l:
                self.scaling *= 1 / self.SCALING_FACTOR
            if event.key == pygame.K_o:
                self.scaling *= self.SCALING_FACTOR
            if event.key == pygame.K_m:
                self.centering_position[0] -= self.MOVING_FACTOR
            if event.key == pygame.K_k:
                self.centering_position[0] += self.MOVING_FACTOR
