from pg_config.pg_space import PgBoxSpace, PgDiscreteSpace, PgConstantSpace


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
        Parameter.vehicle_length: PgConstantSpace(4.0),
        Parameter.vehicle_width: PgConstantSpace(1.5),
        Parameter.vehicle_height: PgConstantSpace(1),
        Parameter.chassis_height: PgConstantSpace(0.3),
        Parameter.front_tire_longitude: PgConstantSpace(1.05),
        Parameter.rear_tire_longitude: PgConstantSpace(1.17),
        Parameter.tire_lateral: PgConstantSpace(0.8),
        Parameter.tire_radius: PgConstantSpace(0.25),
        Parameter.mass: PgConstantSpace(800.0),
        Parameter.heading: PgConstantSpace(0.0),

        # visualization
        Parameter.vehicle_vis_h: PgConstantSpace(180),
        Parameter.vehicle_vis_y: PgConstantSpace(0.1),
        Parameter.vehicle_vis_z: PgConstantSpace(-0.31),
        Parameter.vehicle_vis_scale: PgConstantSpace(0.013),

        # TODO the following parameters will be opened soon using PgBoxSPace
        Parameter.steering_max: PgConstantSpace(40.0),
        Parameter.engine_force_max: PgConstantSpace(500.0),
        Parameter.brake_force_max: PgConstantSpace(40.0),
        Parameter.speed_max: PgConstantSpace(120),
    }


class BlockParameterSpace:
    """
    Make sure the range of curve parameters covers the parameter space of other blocks,
    otherwise, an error may happen in navigation info normalization
    """
    STRAIGHT = {Parameter.length: PgBoxSpace(min=20.0, max=100.0)}
    CURVE = {
        Parameter.length: PgBoxSpace(min=20.0, max=50.0),
        Parameter.radius: PgBoxSpace(min=25.0, max=60.0),
        Parameter.angle: PgBoxSpace(min=30, max=180),
        Parameter.dir: PgDiscreteSpace(2)
    }
    INTERSECTION = {
        Parameter.radius: PgBoxSpace(min=5, max=15),
        Parameter.change_lane_num: PgDiscreteSpace(number=2),  # 0,1
        Parameter.decrease_increase: PgDiscreteSpace(number=2)  # 0, decrease, 1 increase
    }
    ROUNDABOUT = {
        Parameter.radius_exit: PgBoxSpace(min=5, max=15),
        Parameter.radius_inner: PgBoxSpace(min=15, max=45),
        Parameter.angle: PgBoxSpace(min=60, max=70)
    }
    T_INTERSECTION = {
        Parameter.radius: PgBoxSpace(min=5, max=15),
        Parameter.t_intersection_type: PgDiscreteSpace(number=3),  # 3 different t type for previous socket
        Parameter.change_lane_num: PgDiscreteSpace(2),  # 0,1
        Parameter.decrease_increase: PgDiscreteSpace(2)  # 0, decrease, 1 increase
    }
    RAMP_PARAMETER = {
        Parameter.length: PgBoxSpace(min=20, max=40)  # accelerate/decelerate part length
    }
    FORK_PARAMETER = {
        Parameter.length: PgBoxSpace(min=20, max=40),  # accelerate/decelerate part length
        Parameter.lane_num: PgDiscreteSpace(2)
    }
