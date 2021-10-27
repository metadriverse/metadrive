from metadrive.component.block.base_block import BaseBlock
from metadrive.component.map.base_map import BaseMap


def is_map_related_instance(obj):
    return True if isinstance(obj, BaseBlock) or isinstance(obj, BaseMap) else False


def is_map_related_class(object_class):
    return True if issubclass(object_class, BaseBlock) or issubclass(object_class, BaseMap) else False
