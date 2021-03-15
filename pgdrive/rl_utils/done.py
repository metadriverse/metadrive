import logging
from pgdrive.scene_creator.vehicle.base_vehicle import BaseVehicle


def pg_done_function(vehicle: BaseVehicle) -> float:
    done = False
    done_info = dict(crash_vehicle=False, crash_object=False, out_of_road=False, arrive_dest=False)
    long, lat = vehicle.routing_localization.final_lane.local_coordinates(vehicle.position)

    if vehicle.routing_localization.final_lane.length - 5 < long < vehicle.routing_localization.final_lane.length + 5 \
            and vehicle.routing_localization.map.lane_width / 2 >= lat >= (
            0.5 - vehicle.routing_localization.map.lane_num) * vehicle.routing_localization.map.lane_width:
        done = True
        logging.info("Episode ended! Reason: arrive_dest.")
        done_info["arrive_dest"] = True
    elif vehicle.crash_vehicle:
        done = True
        logging.info("Episode ended! Reason: crash. ")
        done_info["crash_vehicle"] = True
    elif vehicle.out_of_route or not vehicle.on_lane or vehicle.crash_side_walk:
        done = True
        logging.info("Episode ended! Reason: out_of_road.")
        done_info["out_of_road"] = True
    elif vehicle.crash_object:
        done = True
        done_info["crash_object"] = True

    # for compatibility
    # crash almost equals to crashing with vehicles
    done_info["crash"] = done_info["crash_vehicle"] or done_info["crash_object"]
    vehicle.step_info.update(done_info)
    return done
