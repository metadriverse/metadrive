.. _config_system:

##########################
Config System
##########################

A MetaDrive instance accepts a dict as the environmental config. For example, you can build a MetaDrive instance with 200 generated maps via

.. code-block:: python

    from metadrive import MetaDriveEnv
    config = dict(environment_num=200, start_seed=0)
    env = MetaDriveEnv(config)

    # or using gym interface:
    import gym
    env = gym.make("MetaDrive-v0", config=config)

In this page, we describe the details of each configurable options.


Generalization Config
########################

MetaDrive can generate unlimited driving scenarios if using procedural generation.
We can specify the range of the scenarios used in different tasks.
For example, you can use different set of generated scenarios to train and test the trained agents.
To achieve that, you only need to specify the range of random seeds used to generated those scenarios.
Concretely, MetaDrive will use the seeds in range :code:`[start_seed, start_seed + environment_num)`.
Therefore, you only need to specify these two values in the config:

    - :code:`start_seed` (int = 0): random seed of the first map
    - :code:`environment_num` (int = 1): number of the driving scenarios



Map Config
#############

MetaDrive provides detailed configuration on the generated maps. Generally speaking, we allow two forms of map generation if using Procedural Generation (PG) algorithm while not loading map from dataset:

1. :code:`config["map_config"]["type"] = "block_num"`: the user specifies the number of blocks in each map so that the PG algorithm will automatically build maps containing that number of blocks while randomizing all parameters including the type of blocks.
2. :code:`config["map_config"]["type"] = "block_sequence"`: the user specify the sequence of block types and PG algorithm will build maps strictly following that order while randomizing the parameters in each block.

We describe all optional map config as follows:

    - :code:`random_lane_width` (bool = False): whether to randomize the width of lane in each map (all lanes in the same map share the same lane width)
    - :code:`random_lane_num` (bool = False): whether to randomize the number of lane in a road in each map (all road in the same map share the same number of lanes)
    - :code:`map_config` (dict): A nested dict describing the generation of map.
        - :code:`type` (str = "block_num"): A string in ["block_num", "block_sequence"] denoting which form of map generation should the PG algorithm use.
        - :code:`config`: XXX
        - :code:`lane_width` (float = 3.5): the width of each lane. This will be overwritten if :code:`random_lane_width = True`.
        - :code:`lane_num` (int = 3): number of lanes in each road. This will be overwritten if :code:`random_lane_num = True`.
        - :code:`exit_length` (float = 50): more than one exit whose length is *exit_length* are contained in some blocks like roundabout


We also provide a shortcut to specify the map:

    -   :code:`map` (int or string): User can set a *string* or *int* as the key to generate map in an easy way. For example, :code:`config["map"] = 3` means generating a map containing 3 blocks, while :code:`config["map"] = "SCrRX"` means the first block is Straight, and the following blocks are Circular, InRamp, OutRamp and Intersection. The character here are the unique ID of different types of blocks as shown in the next table. Therefore using a *string* can determine the block type sequence.
        We provide the following block types:

            +---------------+-----------+
            | Block Type    |    ID     |
            +===============+===========+
            | Straight      |     S     |
            +---------------+-----------+
            | Circular      |     C     |
            +---------------+-----------+
            | InRamp        |     r     |
            +---------------+-----------+
            | OutRamp       |     R     |
            +---------------+-----------+
            | Roundabout    |     O     |
            +---------------+-----------+
            | Intersection  |     X     |
            +---------------+-----------+
            | TIntersection |     T     |
            +---------------+-----------+
            | Merge         |     y     |
            +---------------+-----------+
            | Split         |     Y     |
            +---------------+-----------+
            | Tollgate      |     $     |
            +---------------+-----------+
            | Parking-lot   |     P     |
            +---------------+-----------+
            | Fork          |    WIP    |
            +---------------+-----------+






Action Config
##############

    - :code:`manual_control` (bool = False): whether to control ego vehicle by user in the interface (require :code:`use_render = True`)
    - :code:`controller` (str = "keyboard"): select in ["keyboard", "joystick"], the controller for user to control the ego vehicle
    - :code:`discrete_action` (bool = False): whether to discretize the action space
    - :code:`discrete_steering/throttle_dim` (int = 5, 5): how many dimensions used to discrtize the action space
    - :code:`decision_repeat` (int): how many times for the simulation engine to repeat the applied action to the vehicles. The minimal simulation interval :code:`physics_world_step_size` is 0.02 s. Therefore each RL step will last :code:`decision_repeat * 0.02 s` in the simulation world.



Agent Config
#############

    - :code:`random_agent_model` (bool = False): whether to randomize the dynamic model of ego vehicle
    - :code:`IDM_agent` (bool = False): whether to control ego vehicle by IDM policy





Visualization & Rendering Config
##################################

The config in this part specifies the setting related to visualization. The :code:`use_render` is the most useful one.

    - :code:`use_render` (bool = False): whether to pop a window on your screen or not. This is irrelevant to the vision-based observation.
    - :code:`disable_model_compression` (bool = True): Model compression reduces the memory consumption when using Panda3D window to visualize. Disabling model compression greatly improves the launch speed but might cause breakdown in low-memory machine.
    - :code:`cull_scene` (bool = True): When you want to access the image of camera, it should be set to True.
    - :code:`use_chase_camera_follow_lane` (bool = False): whether to force the third-person view camera following the heading of current lane
    - :code:`camera_dist/height` (float = 6.0, 1.8): the initial distance and height of the third-person view camera
    - :code:`prefer_track_agent` (str = None): specify the name of the agent that you wish to track in the third-person view. This is useful in the visualization in multi-agent environments.
    - :code:`draw_map_resolution` (int = 1024): the size of the image capturing the top-down view of the road network
    - :code:`top_down_camera_initial_x/y/z` (int = 0, 0, 200): the initial position of the top-down view camera


Vehicle Config
################

We list the vehicle config here. Observation Space will be adjusted by these config automatically. For example, if you set :code:`config["vehicle_config"]["lidar"]["num_lasers"] = 720`, then the dimension of the Lidar observation will automatically set to 720.

    - :code:`vehicle_config` (dict):
        - :code:`lidar` (dict): the config is related to the :ref:`Lidar-like observation <State Vector>`. This Lidar only scans nearby vehicles.
            - :code:`num_lasers` (int = 240): the number of lasers used in Lidar
            - :code:`distance` (float = 50.0): the perception field radius
            - :code:`num_others` (int = 0): if this is greater than 0, MetaDrive will retrieve the states of :code:`num_others`-nearest vehicles as additional information
            - :code:`gaussian_noise` (float = 0.0): if this is greater than 0, MetaDrive will add Gaussian noise with :code:`gaussian_noise` standard deviation to each entry of the Lidar cloud points
            - :code:`dropout_prob` (float = 0.0): in [0, 1]. If this is greater than 0, MetaDrive will randomly set :code:`dropout_prob` % of entries in the cloud points to zero
        - :code:`side_detector` (dict): This Lidar only scans the side of the road but not vehicles. The config dict has identical keys as :code:`lidar` except :code:`num_others`.
        - :code:`lane_line_detector` (dict): This Lidar only scans the side of current lane but neither vehicles or road boundary. The config dict has identical keys as :code:`lidar` except :code:`num_others`.
        - :code:`show_lidar` (bool = False): whether to show the end of each Lidar laser in the scene
        - :code:`increment_steering` (bool = False): for keyboard control. When set to True, the steering angle and acceleration is determined by the key pressing time
        - :code:`vehicle_model` (str = "default"): which type of vehicle to use in ego vehicle (s, m, l, xl, default)
        - :code:`enable_reverse` (bool = False): If True and vehicle speed < 0, a brake action (e.g. acceleration = -1) will be parsed as reverse. This is used in the Multi-agent Parking Lot environment.
        - :code:`extra_action_dim` (int = 0): If you want to input more control signal than the default [steering, throttle/brake] in your customized environment, change the default value 0 to the extra number of dimensions.
        - :code:`random_color` (bool = False): whether to randomize the color of ego vehicles. This is useful in multi-agent environments.
        - :code:`image_source` (str = "rgb_camera"): select in ["rgb_camera", "depth_camera"]. When using image observation, it decides where the image collected. See :ref:`use_native_rendering` for more information.
        - :code:`rgb_camera` (tuple = (84, 84): (camera resolution width (int), camera resolution height (int). We use (84, 84) as the default size so that the RGB observation is compatible to those CNN used in Atari. Please refer to :ref:`use_native_rendering` for more information about using image as observation.
        - :code:`spawn_lane_index` (tuple): which lane to spawn this vehicle. Default to one lane in the first block of the map
        - :code:`spawn_longitude/lateral` (float = 5.0, 0.0): The spawn point will be calculated by *spawn_longitude* and *spawn_lateral*
        - :code:`destination` (str = None): the destination road node name. This is used in real dataset replay map.
        - :code:`mini_map` (tuple): (camera resolution width(int), camera resolution height(int), camera height). The size of the bird-view image in the left upper corner of the interface.





Other Observation Config
##########################

The vehicle config decides many of the observational config.

    - :code:`offscreen_render` (bool = False): If you want to use vision-based observation, please set this to True. See :ref:`use_native_rendering` for more information.
    - :code:`rgb_clip` (bool = True): if True than squeeze the value between \[0, 255\] to \[0.0, 1.0\]
    - :code:`headless_machine_render` (bool = False): Set this to True only when training on headless machine and using rgb image


Traffic Config
##################################


Currently, MetaDrive provides two built-in traffic modes: Respawn mode and Trigger mode.


In Respawn mode, Traffic Manager assigns traffic vehicles to random spawn points on the map.
The vehicles immediately start driving toward their destinations after spawning.
When a traffic vehicle terminates, it will be re-positioned to an available spawn point.
Respawn traffic mode is designed to maintain traffic flow density.

On the contrary, the Trigger mode traffic flow is designed to maximize the interaction between target vehicles and traffic vehicles.
The vehicles stay still in the spawn points until the target agent enters the trigger zone in each block.
Take an Intersection block as an case, the traffic vehicles inside the intersection will be triggered and start moving only when the target vehicle trespasses into the intersection.

Here we provide many config to adjust the traffic flow. Note that you can even setup rule-based traffic flow by setting :code:`traffic_mode` > 0.


    - :code:`traffic_density` (float = 0.1): number of traffic vehicles per 10 meter per lane
    - :code:`traffic_mode` (str = "Trigger"): select in ["Trigger", "Respawn"]
    - :code:`random_traffic` (bool = False): If set to False, each driving scenario will have deterministic traffic flow. Otherwise the traffic generation will not be controlled by current seed and provide various traffic flow even in the same road network.


Multi-agent Config
##################


    - :code:`num_agents` (int = 1): the number of agent that are controllable by RL policies
    - :code:`is_multi_agent` (bool = False): set this to True if in multi-agent training (default to True in MA)
    - :code:`allow_respawn` (bool = False): whether allow (default to True in MA)
    - :code:`delay_done` (int = 0): how many environmental steps for the agent to stay static as an obstacle after it is terminated (default to 25 in MA)
    - :code:`horizon` (int = None): The maximum length of each episode. Set to None to remove constraint. (default to 1000 in MA, see :ref:`Multi-agent Environments`)



Reward, Cost and Termination Function Config
##############################################

There are a lot of coefficients to describe the reward function and cost function.
You can adjust the default reward function or design your own functions.
Please refer to :ref:`Reward Function`, :ref:`Cost Function` and :ref:`Termination Function` for more information.


Engine Config
################

This is the engine core config of MetaDrive, including physics engine, window size and so on.
We don't suggest to modify this part if you are not confident on what you are doing.

    - :code:`window_size` (tuple): width and height of interface window. Default is (1200, 900).
    - :code:`physics_world_step_size` (float = 0.02): the minimum time interval between two time steps of bullet physics engine.
    - :code:`show_fps` (bool = True): Turn on/ turn off the frame rater.
    - :code:`debug_physics_world` (bool = False): if True then only render physics world without model
    - :code:`debug_static_world` (bool = True): if True then merge the static world and dynamic world to one world and render this world
    - :code:`pstats` (bool = False): if True then use Panda3D built-in debug tool to profile the program
    - :code:`global_light` (bool = False): True to enable global light. It will consume more computation resource to render.
    - :code:`debug` (bool = False): for developing use, draw the scene with bounding box


Default Config
################

The default config dicts are widely spread in many files. The basic config about some general setting is provided in the `BaseEnv Class <https://github.com/metadriverse/metadrive/blob/main/metadrive/envs/base_env.py>`_.
More detailed config is provided in the `MetaDriveEnv Class <https://github.com/metadriverse/metadrive/blob/main/metadrive/envs/metadrive_env.py>`_.
Besides, for `SafeMetaDriveEnv Class <https://github.com/metadriverse/metadrive/blob/main/metadrive/envs/safe_metadrive_env.py>`_
and `MultiAgentMetaDrive Class <https://github.com/metadriverse/metadrive/blob/main/metadrive/envs/marl_envs/multi_agent_metadrive.py>`_
there also have many task-specified config. Please feel free to open issues if you have any question about the environmental settings!



