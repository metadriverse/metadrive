from pgdrive.pg_config.pg_space import PGBoxSpace, PGDiscreteSpace, PGConstantSpace


class Parameter:
    """
    Block parameters and vehicle parameters
    """
    # block
    length = "length"
    radius = "radius"
    angle = "angle"
    goal = "goal"
    dir = "dir"
    radius_inner = "inner_radius"  # only for roundabout use
    radius_exit = "exit_radius"
    t_intersection_type = "t_type"
    lane_num = "lane_num"
    change_lane_num = "change_lane_num"
    decrease_increase = "decrease_increase"

    # vehicle
    vehicle_length = "v_len"
    vehicle_width = "v_width"
    vehicle_height = "v_height"
    front_tire_longitude = "f_tire_long"
    rear_tire_longitude = "r_tire_long"
    tire_lateral = "tire_lateral"
    tire_axis_height = "tire_axis_height"
    tire_radius = "tire_radius"
    mass = "mass"
    chassis_height = "chassis_height"
    heading = "heading"
    steering_max = "steering_max"
    engine_force_max = "e_f_max"
    brake_force_max = "b_f_max"
    speed_max = "s_max"

    # vehicle visualization
    vehicle_vis_z = "vis_z"
    vehicle_vis_y = "vis_y"
    vehicle_vis_h = "vis_h"
    vehicle_vis_scale = "vis_scale"


class VehicleParameterSpace:
    BASE_VEHICLE = {
        # Now the parameter sample is not available and thus the value space is incorrect
        Parameter.vehicle_length: PGConstantSpace(4.0),
        Parameter.vehicle_width: PGConstantSpace(1.5),
        Parameter.vehicle_height: PGConstantSpace(1),
        Parameter.chassis_height: PGConstantSpace(0.3),
        Parameter.front_tire_longitude: PGConstantSpace(1.05),
        Parameter.rear_tire_longitude: PGConstantSpace(1.17),
        Parameter.tire_lateral: PGConstantSpace(0.8),
        Parameter.tire_radius: PGConstantSpace(0.25),
        Parameter.mass: PGConstantSpace(800.0),
        Parameter.heading: PGConstantSpace(0.0),

        # visualization
        Parameter.vehicle_vis_h: PGConstantSpace(180),
        Parameter.vehicle_vis_y: PGConstantSpace(0.1),
        Parameter.vehicle_vis_z: PGConstantSpace(-0.31),
        Parameter.vehicle_vis_scale: PGConstantSpace(0.013),

        # TODO the following parameters will be opened soon using PGBoxSPace
        Parameter.steering_max: PGConstantSpace(40.0),
        Parameter.engine_force_max: PGConstantSpace(500.0),
        Parameter.brake_force_max: PGConstantSpace(40.0),
        Parameter.speed_max: PGConstantSpace(120),
    }


class BlockParameterSpace:
    """
    Make sure the range of curve parameters covers the parameter space of other blocks,
    otherwise, an error may happen in navigation info normalization
    """
    STRAIGHT = {Parameter.length: PGBoxSpace(min=40.0, max=80.0)}
    CURVE = {
        Parameter.length: PGBoxSpace(min=40.0, max=80.0),
        Parameter.radius: PGBoxSpace(min=25.0, max=60.0),
        Parameter.angle: PGBoxSpace(min=45, max=135),
        Parameter.dir: PGDiscreteSpace(2)
    }
    INTERSECTION = {
        Parameter.radius: PGConstantSpace(10),
        Parameter.change_lane_num: PGDiscreteSpace(number=2),  # 0, 1
        Parameter.decrease_increase: PGDiscreteSpace(number=2)  # 0, decrease, 1 increase
    }
    ROUNDABOUT = {
        Parameter.radius_exit: PGBoxSpace(min=5, max=15),  # TODO Should we reduce this?
        Parameter.radius_inner: PGBoxSpace(min=15, max=45),  # TODO Should we reduce this?
        Parameter.angle: PGConstantSpace(60)
    }
    T_INTERSECTION = {
        Parameter.radius: PGConstantSpace(10),
        Parameter.t_intersection_type: PGDiscreteSpace(number=3),  # 3 different t type for previous socket
        Parameter.change_lane_num: PGDiscreteSpace(2),  # 0,1
        Parameter.decrease_increase: PGDiscreteSpace(2)  # 0, decrease, 1 increase
    }
    RAMP_PARAMETER = {
        Parameter.length: PGBoxSpace(min=20, max=40)  # accelerate/decelerate part length
    }
    FORK_PARAMETER = {
        Parameter.length: PGBoxSpace(min=20, max=40),  # accelerate/decelerate part length
        Parameter.lane_num: PGDiscreteSpace(2)
    }
