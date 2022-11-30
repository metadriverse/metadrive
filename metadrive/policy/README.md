Here we define a lot of policies that can be plugged in to control arbitrary vehicles.

They are:

* IDMPolicy(BasePolicy): A heuristic rule-based policy.

* ManualControllableIDMPolicy(IDMPolicy): If human is not taking over, then use IDM policy.

* WaymoIDMPolicy(IDMPolicy): A better rule-based policy for the traffic car in Waymo environment. (not for ego car) 

* EgoWaymoIDMPolicy(IDMPolicy): This policy is customized for the ego car in Waymo environment.

* ReplayEgoCarPolicy(BasePolicy): Make the ego car replay the logged trajectory.

Change the `env_config["agent_policy"]` to `IDMPolicy|EgoWaymoIDMPolicy|ReplayEgoCarPolicy` to let the ego car follow different policies.

