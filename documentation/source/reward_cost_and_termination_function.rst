###############################################
Reward, Cost, Termination and Step Information
###############################################

Following the standard OpenAI Gym API, after each step of the environment :code:`env.step(...)`, the environment will return
a tuple containing four items: :code:`(obs, reward, done, info)`. In this page, we discuss the design of reward function :code:`reward`, cost function :code:`info["cost"]`,
termination criterion :code:`done` in various settings and the details of step information :code:`info`.

Reward Function
#################

The default reward function in MetaDrive only contains a dense driving reward and a sparse terminal reward. The dense reward is the longitudinal movement toward destination in Frenet coordinates.
The sparse reward :math:`+20` is given when the agent arrives the destination.


However, MetaDrive actually prepares a complex reward function that enables user to customize their reward functions from config dict directly.
The complete reward function is composed of four parts as follows:

.. math::

    R = c_{1} R_{driving} + c_{2} R_{speed} + R_{termination}




- The **driving reward**  :math:`R_{driving} = d_t - d_{t-1}`, wherein the :math:`d_t` and :math:`d_{t-1}` denote the longitudinal coordinates of the target vehicle in the current lane of two consecutive time steps, providing dense reward to encourage agent to move forward.
- The **speed reward** :math:`R_{speed} = v_t/v_{max}` incentives agent to drive fast. :math:`v_{t}` and :math:`v_{max}` denote the current velocity and the maximum velocity (80 km/h), respectively.
- The **termination reward** :math:`R_{termination}` contains a set of sparse rewards. At the end of episode, other dense rewards will be disabled and only one sparse reward will be given to the agent at the end of the episode according to its termination state. We implement the :code:`success_reward`, :code:`out_of_road_penalty`, :code:`crash_vehicle_penalty` and :code:`crash_object_penalty` currently. The penalty will be given as negative reward.

We also provide a config call :code:`use_lateral_reward`, which is a multiplier in range [0, 1] indicating whether the ego vehicle is far from the center of current lane. The multiplier will apply to the driving reward.

We summarize the default reward config here:


- :code:`success_reward = 10.0`: one of termination reward.
- :code:`out_of_road_penalty = 5.0`: will use -5.0 as the termination reward.
- :code:`crash_vehicle_penalty = 5.0`: will use -5.0 as the termination reward.
- :code:`crash_object_penalty = 5.0`: will use -5.0 as the termination reward.
- :code:`driving_reward = 1.0`: the :math:`c_{1}` in reward function.
- :code:`speed_reward = 0.1`: the :math:`c_{2}` in reward function.
- :code:`use_lateral_reward = False`: disable weighting the driving reward according to centering in the lane.

The reward function is implemented in the :code:`reward_function` in `MetaDriveEnv <https://github.com/metadriverse/metadrive/blob/main/metadrive/envs/metadrive_env.py#L209>`_.


Cost Function
#################

Similar to the reward function, we also provide default cost function to measure the safety during driving. The cost function will be placed in the returned information dict as :code:`info["cost"]` after :code:`env.step` function.

- :code:`crash_vehicle_cost = 1.0`: yield cost when crashing to other vehicles.
- :code:`crash_object_cost = 1.0`: yield cost when crashing to objects, such as cones and triangles.
- :code:`out_of_road_cost = 1.0`: yield cost when driving out of the road.

The cost function is implemented in the :code:`cost_function` in `MetaDriveEnv <https://github.com/metadriverse/metadrive/blob/main/metadrive/envs/metadrive_env.py#L188>`_.

Termination Function
#######################

MetaDrive will terminate an episode of a vehicle if:

1. the target vehicle arrive its destination,
2. the vehicle drives out of the road,
3. the vehicle crashes to other vehicles,
4. the vehicle crashes to obstacles, or
5. the vehicle crashes to building (e.g. in Multi-agent Tollgate environment).

The above termination criterion is implemented in the :code:`done_function` in `MetaDriveEnv <https://github.com/metadriverse/metadrive/blob/main/metadrive/envs/metadrive_env.py#L153>`_.

Please note that in the Safe RL environment `SafeMetaDriveEnv <https://github.com/metadriverse/metadrive/blob/main/metadrive/envs/safe_metadrive_env.py>`_, the episode will not be terminated when vehicles crashing into objects or vehicles.
This is because we wish to investigate the safety performance of a vehicle in an extremely dangerous environments.
Terminating episodes too frequently will let the training becomes too hard to complete.

In Multi-agent environment, the above termination criterion is still hold for each vehicle. We call this the termination of an *agent episode*.
We explicitly add two config to adjust the termination processing in MARL: :code:`crash_done = True` and :code:`out_of_road_done = True`.
They denotes whether to terminate the agent episode if crash / out of road happens.

Besides, in Multi-agent environment, the controllable target vehicles consistently respawn in the scene if old target vehicles are terminated.
To limit the length of *environmental episode*, we also introduce a config :code:`horizon = 1000` in MARL environments.
The environmental episode has a **minimal length** of :code:`horizon` steps and the environment will stop spawning new target vehicles if this horizon is exceeded.
If you wish to disable the respawning mechanism in MARL, set the config :code:`allow_respawn = False`. In this case, the environmental episode will terminate if no active vehicles are in the scene.


Step Information
#######################

The step information dict :code:`info` contains rich information about current state of the environment and the target vehicle.
We summarize the dict as follows:

.. code-block::

    {
        # Number of vehicles being overtaken by ego vehicle in this episode
        'overtake_vehicle_num': 0,

        # Current velocity in km/h
        'velocity': 0.0,

        # The current normalized steering signal in [-1, 1]
        'steering': -0.06901532411575317,

        # The current normalized acceleration signal in [-1, 1]
        'acceleration': -0.2931942343711853,

        # The normalized action after clipped who is applied to the ego vehicle
        'raw_action': (-0.06901532411575317, -0.2931942343711853),

        # Whether crash to vehicle / object / building
        'crash_vehicle': False,
        'crash_object': False,
        'crash_building': False,
        'crash': False,  # Whether any kind of crash happens

        # Whether going out of the road / arrive destination
        # or exceeding the maximal episode length
        'out_of_road': False,
        'arrive_dest': False,
        'max_step': False,

        # The reward in this time step / the whole episode so far
        'step_reward': 0.0,
        'episode_reward': 0.0,

        # The cost in this time step
        'cost': 0,

        # The length of current episode
        'episode_length': 1
    }

In Safe RL environment `SafeMetaDriveEnv <https://github.com/metadriverse/metadrive/blob/main/metadrive/envs/safe_metadrive_env.py>`_, we additionally record the :code:`info["total_cost"]` to record the sum of all cost within one episode.

The step info is collected from various sources such as the engine, reward function, termination function, traffic manager, agent manager and so on.




