import logging

from pgdrive.constants import BodyName


def pg_collision_callback(contact):
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
        if nodes[i].hasPythonTag(BodyName.Ego_vehicle):
            another_node_name = another_nodes[i].getName()
            if another_node_name in [BodyName.Traffic_vehicle, BodyName.Ego_vehicle]:
                nodes[i].getPythonTag(BodyName.Ego_vehicle).crash_vehicle = True
            elif another_node_name in [BodyName.Traffic_cone, BodyName.Traffic_triangle]:
                nodes[i].getPythonTag(BodyName.Ego_vehicle).crash_object = True
            # TODO update this
            # self._frame_objects_crashed.append(node.getPythonTag(name[0]))
            logging.debug("{} crash with {}".format(nodes[i].getName(), another_nodes[i].getName()))
