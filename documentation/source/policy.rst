.. _policy:

######################
Policy Classes
######################

We define a lot of policies that can be plugged in to control arbitrary vehicles.

They are:

* IDMPolicy(BasePolicy): A heuristic rule-based policy.

* ManualControllableIDMPolicy(IDMPolicy): If human is not taking over, then use IDM policy.

* WaymoIDMPolicy(IDMPolicy): A better rule-based policy for the traffic car in Waymo environment. (including the ego car)

* ReplayEgoCarPolicy(BasePolicy): Make the ego car replay the logged trajectory.

Change the `env_config["agent_policy"]` to `IDMPolicy|WaymoIDMPolicy|ReplayEgoCarPolicy` to let the ego car follow different policies.

