from metadrive.constants import MetaDriveType
from metadrive.utils.utils import get_object_from_node


def collision_callback(contact):
    """
    All collision callback should be here, and a notify() method can turn it on
    It may lower the performance if overdone
    """

    # now it only process BaseVehicle collision
    node0 = contact.getNode0()
    node1 = contact.getNode1()

    nodes = [node0, node1]
    another_nodes = [node1, node0]
    for i in range(2):
        if nodes[i].hasPythonTag(MetaDriveType.VEHICLE):
            obj_type = another_node_name = another_nodes[i].getName()
            if obj_type in [MetaDriveType.BOUNDARY_SIDEWALK, MetaDriveType.CROSSWALK] \
                    or MetaDriveType.is_road_line(obj_type):
                continue
            # print(obj_type)
            obj_1 = get_object_from_node(nodes[i])
            obj_2 = get_object_from_node(another_nodes[i])

            # crash vehicles
            if another_node_name == MetaDriveType.VEHICLE:
                obj_1.crash_vehicle = True
            # crash objects
            elif MetaDriveType.is_traffic_object(another_node_name):
                if not obj_2.crashed:
                    obj_1.crash_object = True
                    if obj_2.COST_ONCE:
                        obj_2.crashed = True
            # collision_human
            elif another_node_name in [MetaDriveType.CYCLIST, MetaDriveType.PEDESTRIAN]:
                obj_1.crash_human = True
            # crash invisible wall or building
            elif another_node_name in [MetaDriveType.INVISIBLE_WALL, MetaDriveType.BUILDING]:
                obj_1.crash_building = True
            # logging.debug("{} crash with {}".format(nodes[i].getName(), another_nodes[i].getName()))
