from panda3d.bullet import BulletWorld


class CollisionGroup:
    Terrain = 2
    EgoVehicle = 1
    EgoVehicleTop = 6
    LaneLine = 3
    TrafficVehicle = 4
    LaneSurface = 5  # useless now, since it is in another physics world

    @classmethod
    def collision_rules(cls):
        return [
            # terrain collision
            (cls.Terrain, cls.Terrain, False),
            (cls.Terrain, cls.LaneLine, False),
            (cls.Terrain, cls.LaneSurface, False),
            (cls.Terrain, cls.EgoVehicle, True),
            # change it after we design a new traffic system !
            (cls.Terrain, cls.TrafficVehicle, False),

            # block collision
            (cls.LaneLine, cls.LaneLine, False),
            (cls.LaneLine, cls.LaneSurface, False),
            (cls.LaneLine, cls.EgoVehicle, True),
            # change it after we design a new traffic system !
            (cls.LaneLine, cls.TrafficVehicle, False),

            # traffic vehicles collision
            (cls.TrafficVehicle, cls.TrafficVehicle, False),
            (cls.TrafficVehicle, cls.LaneSurface, False),
            (cls.TrafficVehicle, cls.EgoVehicle, True),

            # ego vehicle collision
            (cls.EgoVehicle, cls.EgoVehicle, True),
            (cls.EgoVehicle, cls.EgoVehicleTop, False),
            (cls.EgoVehicle, cls.LaneSurface, False),

            # lane surface
            (cls.LaneSurface, cls.LaneSurface, False),
        ]

    @classmethod
    def set_collision_rule(cls, dynamic_world: BulletWorld):
        for rule in cls.collision_rules():
            dynamic_world.setGroupCollisionFlag(*rule)
