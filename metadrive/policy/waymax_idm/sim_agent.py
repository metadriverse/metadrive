# Copyright 2023 The Waymax Authors.
#
# Licensed under the Waymax License Agreement for Non-commercial Use
# Use (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     https://github.com/waymo-research/waymax/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base class for Waymax sim agents."""
import abc
from typing import Any, Callable, Optional

import jax

from metadrive.policy.waymax_idm import actor_core
from metadrive.policy.waymax_idm import datatypes

_DEFAULT_CONTROL_FUNC = lambda state: ~state.object_metadata.is_sdc


@actor_core.register_actor_core
class SimAgentActor(actor_core.WaymaxActorCore):
    """Base class for simulated agents.

    Subclasses should implement the `update_trajectory` method. As SimAgentActor
    outputs TrajectoryUpdate actions, it is primarily intended to be used with
    the StateDynamics dynamics model.
    """

    def __init__(
            self,
            is_controlled_func: Optional[
                Callable[[datatypes.SimulatorState], jax.Array]
            ] = None,
    ):
        """Constructs a sim agent.

        Args:
          is_controlled_func: A function that maps a SimulatorState to a boolean
            mask of shape (..., num_objects) indicating which objects are to be
            controlled by this sim agent.
        """
        super().__init__()
        if is_controlled_func is None:
            is_controlled_func = _DEFAULT_CONTROL_FUNC
        self.is_controlled_func = is_controlled_func

    @abc.abstractmethod
    def update_trajectory(
            self, state: datatypes.SimulatorState
    ) -> datatypes.TrajectoryUpdate:
        """Updates the trajectory for all simulated agents.

        Args:
          state: The current simulator state.

        Returns:
          A trajectory update of shape (..., num_objects, num_timesteps=1) that
          contains the updated positions and velocities for all simulated agents
          for the next timestep.
        """

    def init(self, rng: jax.Array, state: datatypes.SimulatorState):
        """Returns an empty initial state."""

    def select_action(
            self,
            params: actor_core.Params,
            state: datatypes.SimulatorState,
            actor_state: Any,
            rng: jax.Array,
    ) -> actor_core.WaymaxActorOutput:
        """Selects an action given the current simulator state.

        Args:
          params: Actor parameters, e.g. neural network weights.
          state: The current simulator state.
          actor_state: The actor state, e.g. recurrent state or batch normalization.
          rng: A random key.

        Returns:
          An actor output containing the next action and actor state.
        """
        del params, actor_state, rng  # Unused.
        action = self.update_trajectory(state).as_action()
        return actor_core.WaymaxActorOutput(
            action=action,
            actor_state=None,
            is_controlled=self.is_controlled_func(state),
        )

    @property
    def name(self) -> str:
        """Name of the agent used for inspection and logging."""
        return self.__class__.__name__
