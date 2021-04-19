from pgdrive.envs.generation_envs.side_pass_env import SidePassEnv
from pgdrive.constants import BodyName


def test_object_collision_detection(render=False):
    env = SidePassEnv(
        {
            "manual_control": render,
            "use_render": render,
            "debug": False,
            "vehicle_config": {
                "show_lidar": True
            }
        }
    )
    try:
        o = env.reset()
        lane_index = (">>", ">>>", 0)
        alert = env.scene_manager.objects_mgr.spawn_one_object(
            "Traffic Triangle", env.current_map.road_network.get_lane(lane_index), lane_index, 22, 0
        )
        env.alert.attach_to_pg_world(env.pg_world.pbr_worldNP, env.pg_world.physics_world)
        lane_index = (">>", ">>>", 2)
        alert = env.scene_manager.objects_mgr.spawn_one_object(
            BodyName.Traffic_cone, env.current_map.road_network.get_lane(lane_index), lane_index, 22, 0
        )
        env.alert.attach_to_pg_world(env.pg_world.pbr_worldNP, env.pg_world.physics_world)
        crash_obj = False
        detect_obj = False
        for i in range(1, 100000 if render else 2000):
            o, r, d, info = env.step([0, 1])
            for obj in env.vehicle.lidar.get_detected_objects():
                if obj.getNode().hasPythonTag(BodyName.Traffic_cone):
                    detect_obj = True
            if render:
                env.render()
            if info["crash_object"]:
                crash_obj = True
                break
        assert crash_obj and detect_obj, "Can not crash with object!"
    finally:
        env.close()


if __name__ == "__main__":
    test_object_collision_detection(render=True)
