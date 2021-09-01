.. _config_system:

##########################
Environment Configuration
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


Agent Config
#############


    - :code:`num_agents` (int = 1):
    - :code:`is_multi_agent` (bool = False):
    - :code:`allow_respawn` (bool = False):
    - :code:`delay_done` (int = 0):
    - :code:`random_agent_model` (bool = False):
    - :code:`IDM_agent` (bool = False):



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
        - :code:`exit_length` (float = 50): TODO(LQY)


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
            | Fork          |TODO(LQY)  |
            +---------------+-----------+





Action Config
##############

    - :code:`manual_control` (bool = False):
    - :code:`controller` (str = "keyboard"):
    - :code:`decision_repeat` (int = 5):
    - :code:`discrete_action` (bool = False):
    - :code:`discrete_steering/throttle_dim` (int = 5, 5):
    - :code:`decision_repeat` (int): The minimal step size of the world is 2e-2 second, and thus for agent the world will step
      decision_repeat * 2e-2 second after applying one action or step.


Visualization & Rendering
###########################

    - :code:`use_render` (bool = False): Pop a window on your screen or not
    - :code:`debug` (bool = False): For developing use, draw the scene with bounding box
    - :code:`disable_model_compression` (bool = True): Model compression reduces the memory consumption when using Panda3D window to visualize. Disabling model compression greatly improves the launch speed but might cause breakdown in low-memory machine.
    - :code:`cull_scene` (bool = True): When you want to access the image of camera, it should be set to True.
    - :code:`use_chase_camera_follow_lane` (bool = False):
    - :code:`camera_height` (float = 1.8):
    - :code:`camera_dist` (float = 6.0):
    - :code:`prefer_track_agent` (str = None):
    - :code:`draw_map_resolution` (int = 1024):
    - :code:`top_down_camera_initial_x/y/z` (int = 0, 0, 200):


Vehicle Control
#################################

The following content is working in progress.


TrafficManager Config
##################################

    - :code:`traffic_density` (float): Vehicle number per 10 meter, aiming to adjust the number of vehicle on road
    - :code:`traffic_mode`: Trigger mode (Triger) / reborn mode (Reborn). In Reborn mode vehicles will enter the map again after arriving its destination.
    - :code:`random_traffic` (bool): the traffic generation will not be controlled by current map seed. If set to *False*, each map will have same traffic flow.




Observation Config
######################

    - :code:`offscreen_render` (bool): If you want to use camera data, please set this to True.
    - :code:`rgb_clip` (bool): Squeeze the value between \[0, 255\] to \[0.0, 1.0\]
    - :code:`vehicle_config` (dict): Sensor parameters for vehicle
    - :code:`image_source` (str): decided which camera image to use (mini_map or front camera). Now we only support capture one image as a part of
      observation.



Reward Scheme
####################
Coefficient of different kinds of reward to describe the driving goal
Find more information by accessing our source code in MetaDriveEnv
You can adjust our primitive reward function or design your own reward function

Misc.
##########

    - :code:`use_increment_steering` (bool): Keyboard control use discretized action such as -1, 0, +1. You can set this value to True to make the keyboard strokes serve as increments to existing action.
    - :code:`action_check` (bool): Check whether the value of action is between \[0.0, 1.0\] or not.
    - :code:`engine_config` (dict): Some basic settings for low-level physics world. More information can be found in source code.

PGWorld Config
################
    This is the core of MetaDrive, including physics engine, task manager and so on.
     - :code:`window_size` (tuple): Width, height of rendering window.
     - :code:`debug` (bool): The debug value in MetaDriveEnv will be passed to PGWorld.
     - :code:`physics_world_step_size` (float): The minimum step size of bullet physics engine.
     - :code:`show_fps` (bool): Turn on/ turn off the frame rater.
     - :code:`force_fps` (None or float): *None* means no render fps limit, while *float* indicates the maximum render FPS.
     - :code:`decision_repeat` (int): This will be written by MetaDriveEnv to do ForceFPS.
     - :code:`debug_physics_world` (bool): Only render physics world without model, a special debug option.
     - :code:`headless_machine_render` (bool): Set this to true only when training on headless machine and use rgb image!!!!!!
     - :code:`use_render` (bool): The value is same as *use_render* in MetaDriveEnv
     - :code:`offscreen_render` (bool): The value is same as *offscreen_render* in MetaDriveEnv.




Vehicle Config
################

We list the vehicle config here. Observation Space will be adjusted by these config automatically.
Find more information and in our source code and test scripts!

- :code:`lidar` (tuple): (laser num, distance, other vehicle info num)
- :code:`rgb_camera` (tuple): (camera resolution width(int), camera resolution height(int), we use (84, 84) as the default value like what Nature DQN did in Atari.
- :code:`mini_map` (tuple): (camera resolution width(int), camera resolution height(int), camera height). The bird-view image can be captured by this camera.
- :code:`show_navi_mark` (bool): A spinning navigation mark will be shown in the scene
- :code:`increment_steering` (bool): For keyboard control using. When set to True, the steering angle is determined by the key pressing time.
- :code:`wheel_friction` (float): Friction coefficient
