import logging

from pgdrive.scene_creator.vehicle.base_vehicle import BaseVehicle


def pg_done_function(vehicle: BaseVehicle):
    done = False
    done_info = dict(crash_vehicle=False, crash_object=False, out_of_road=False, arrive_dest=False)
    if vehicle.arrive_destination:
        done = True
        logging.info("Episode ended! Reason: arrive_dest.")
        done_info["arrive_dest"] = True
    elif vehicle.crash_vehicle:
        done = True
        logging.info("Episode ended! Reason: crash. ")
        done_info["crash_vehicle"] = True
    elif vehicle.out_of_route or not vehicle.on_lane or vehicle.crash_sidewalk:
        done = True
        logging.info("Episode ended! Reason: out_of_road.")
        done_info["out_of_road"] = True
    elif vehicle.crash_object:
        done = True
        done_info["crash_object"] = True

    # for compatibility
    # crash almost equals to crashing with vehicles
    done_info["crash"] = done_info["crash_vehicle"] or done_info["crash_object"]
    return done, done_info
