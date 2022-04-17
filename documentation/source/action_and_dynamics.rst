.. _action_and_dynamics:

###############
Action Space
###############

MetaDrive receives normalized action as input to control each target vehicle: :math:`\mathbf a = [a_1, a_2]^T \in [-1, 1]^2`.

At each environmental time step, MetaDrive converts the normalized action into the steering :math:`u_s` (degree), acceleration :math:`u_a` (hp) and brake signal :math:`u_b` (hp) in the following ways:


.. math::

    u_s & = S_{max} a_1 ~\\
    u_a & = F_{max} \max(0, a_2) ~\\
    u_b & = -B_{max} \min(0, a_2)

wherein :math:`S_{max}` (degree)  is the maximal steering angle, :math:`F_{max}` (hp) is the maximal engine force, and :math:`B_{max}` (hp) is the maximal brake force.
Since the accurate values of these parameters are varying across different types of vehicle, please refer to the `VehicleParameterSpace Class <https://github.com/metadriverse/metadrive/blob/main/metadrive/utils/space.py#L219>`_ for details.

By such design, the action space for each agent is always fixed to :code:`gym.spaces.Box(low=-1.0, high=1.0, shape=(2, ))`. However, we provides a config named :code:`extra_action_dim` (int) which allows user to add more dimensions in the action space.
For example, if we set :code:`config["extra_action_dim"] = 1`, then the action space for each agent will become :code:`Box(-1.0, 1.0, shape=(3, ))`. This allow the user to write environment wrapper that introduce more input action dimensions.