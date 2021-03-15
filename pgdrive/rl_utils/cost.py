from pgdrive.scene_creator.vehicle.base_vehicle import BaseVehicle


def pg_cost_function(vehicle: BaseVehicle) -> None:
    vehicle.step_info["cost"] = 0
    if vehicle.step_info["crash_vehicle"]:
        vehicle.step_info["cost"] = vehicle.vehicle_config["crash_vehicle_cost"]
    elif vehicle.step_info["crash_object"]:
        vehicle.step_info["cost"] = vehicle.vehicle_config["crash_object_cost"]
    elif vehicle.step_info["out_of_road"]:
        vehicle.step_info["cost"] = vehicle.vehicle_config["out_of_road_cost"]
