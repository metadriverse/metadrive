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

"""Expert actor which returns an action inferred from logged data."""

from typing import Callable

import jax
import jax.numpy as jnp
from metadrive.policy.waymax_idm.state_dynamics import StateDynamics
from metadrive.policy.waymax_idm import actor_core
from metadrive.policy.waymax_idm import datatypes

_EXPERT_NAME = 'expert'
_IS_SDC_FUNC = lambda state: state.object_metadata.is_sdc


def infer_expert_action(
        dynamics_model,
        simulator_state: datatypes.SimulatorState,
) -> datatypes.Action:
    """Infers an action from sim_traj[timestep] to log_traj[timestep + 1].

    Args:
      simulator_state: State of the simulator at the current timestep. Will use
        the `sim_trajectory` and `log_trajectory` fields to calculate an action.
      dynamics_model: Dynamics model whose `inverse` function will be used to
        infer the expert action given the logged states.

    Returns:
      Action that will take the agent from sim_traj[timestep] to
        log_traj[timestep + 1].
    """
    prev_sim_traj = datatypes.dynamic_slice(  # pytype: disable=wrong-arg-types  # jax-ndarray
        simulator_state.sim_trajectory, simulator_state.timestep, 1, axis=-1
    )
    next_logged_traj = datatypes.dynamic_slice(  # pytype: disable=wrong-arg-types  # jax-ndarray
        simulator_state.log_trajectory, simulator_state.timestep + 1, 1, axis=-1
    )
    combined_traj = jax.tree_map(
        lambda x, y: jnp.concatenate([x, y], axis=-1),
        prev_sim_traj,
        next_logged_traj,
    )
    return dynamics_model.inverse(
        combined_traj, metadata=simulator_state.object_metadata, timestep=0
    )


def create_expert_actor(
        is_controlled_func: Callable[
            [datatypes.SimulatorState], jax.Array
        ] = _IS_SDC_FUNC,
) -> actor_core.WaymaxActorCore:
    """Creates an expert agent using the WaymaxActorCore interface.

    This agent infers an action from the `expert` by inferring an action using
    the logged data. It does this by calling the `inverse` function on the passed
    in `dynamics` parameter. It will return an action in the format returned by
    the `dynamics` parameter.

    Args:
      dynamics_model: Dynamics model whose `inverse` function will be used to
        infer the expert action given the logged states.
      is_controlled_func: A function that maps state to a controlled objects mask
        of shape (..., num_objects).

    Returns:
      A Stateless Waymax actor which returns an `expert` action for all controlled
      objects (defined by is_controlled_func) by inferring the best-fit action
      given the logged state.
    """
    dynamics_model = StateDynamics()
    def select_action(  # pytype: disable=annotation-type-mismatch
            params: actor_core.Params,
            state: datatypes.SimulatorState,
            actor_state=None,
            rng: jax.Array = None,
    ) -> actor_core.WaymaxActorOutput:
        """Infers an action for the current timestep given logged state."""
        del params, actor_state, rng  # unused.
        logged_action = infer_expert_action(dynamics_model, state)
        return actor_core.WaymaxActorOutput(
            actor_state=None,
            action=logged_action,
            is_controlled=is_controlled_func(state),
        )

    return actor_core.actor_core_factory(
        init=lambda rng, init_state: None,
        select_action=select_action,
        name=_EXPERT_NAME,
    )
