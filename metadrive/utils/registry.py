_metadrive_class_list = None
_metadrive_class_registry = None


def _initialize_registry():
    global _metadrive_class_list
    _metadrive_class_list = []

    # Register all PG blocks
    from metadrive.component.pgblock.bottleneck import Merge, Split
    from metadrive.component.pgblock.curve import Curve
    from metadrive.component.pgblock.fork import InFork, OutFork
    from metadrive.component.pgblock.parking_lot import ParkingLot
    from metadrive.component.pgblock.ramp import InRampOnStraight, OutRampOnStraight
    from metadrive.component.pgblock.roundabout import Roundabout
    from metadrive.component.pgblock.std_intersection import StdInterSection, StdInterSectionWithUTurn
    from metadrive.component.pgblock.std_t_intersection import StdTInterSection
    from metadrive.component.pgblock.straight import Straight
    from metadrive.component.pgblock.tollgate import TollGate
    from metadrive.component.pgblock.bidirection import Bidirection
    _metadrive_class_list.extend(
        [
            Merge, Split, Curve, InFork, OutFork, ParkingLot, InRampOnStraight, OutRampOnStraight, Roundabout,
            StdInterSection, StdTInterSection, StdInterSectionWithUTurn, Straight, TollGate, Bidirection
        ]
    )

    global _metadrive_class_registry
    _metadrive_class_registry = {k.__name__: k for k in _metadrive_class_list}


def get_metadrive_class(class_name):
    global _metadrive_class_registry
    if _metadrive_class_registry is None:
        _initialize_registry()

    assert class_name in _metadrive_class_registry, "{} is not in Registry: {}".format(
        class_name, _metadrive_class_registry.keys()
    )
    return _metadrive_class_registry[class_name]
