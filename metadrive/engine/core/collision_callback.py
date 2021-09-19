from metadrive.constants import BodyName
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
        if nodes[i].hasPythonTag(BodyName.Vehicle):
            obj_1 = get_object_from_node(nodes[i])
            obj_2 = get_object_from_node(another_nodes[i])
            another_node_name = another_nodes[i].getName()
            # crash vehicles
            if another_node_name == BodyName.Vehicle:
                obj_1.crash_vehicle = True
            # crash objects
            elif another_node_name == BodyName.Traffic_object:
                if not obj_2.crashed:
                    obj_1.crash_object = True
                    if obj_2.COST_ONCE:
                        obj_2.crashed = True
            # crash invisible wall or building
            elif another_node_name in [BodyName.InvisibleWall, BodyName.TollGate]:
                obj_1.crash_building = True
            # logging.debug("{} crash with {}".format(nodes[i].getName(), another_nodes[i].getName()))
