pg_cost_scheme = dict(
    # ===== Cost Scheme =====
    crash_vehicle_cost=1,
    crash_object_cost=1,
    out_of_road_cost=1.
)


def pg_cost_function(vehicle):
    step_info = dict()
    step_info["cost"] = 0
    if vehicle.crash_vehicle:
        step_info["cost"] = vehicle.vehicle_config["crash_vehicle_cost"]
    elif vehicle.crash_object:
        step_info["cost"] = vehicle.vehicle_config["crash_object_cost"]
    elif vehicle.out_of_route:
        step_info["cost"] = vehicle.vehicle_config["out_of_road_cost"]
    return step_info['cost'], step_info
