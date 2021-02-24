from pgdrive.envs.generation_envs.side_pass_env import SidePassEnv

# setup_logger(True)


def test_object_collision(render=False):
    env = SidePassEnv({"manual_control": render, "use_render": render, "debug": True})
    o = env.reset()
    lane_index = (">>", ">>>", 0)
    alert = env.scene_manager.objects_mgr.spawn_one_object(
        "Traffic Triangle", env.current_map.road_network.get_lane(lane_index), lane_index, 22, 0
    )
    env.alert.attach_to_pg_world(env.pg_world.pbr_worldNP, env.pg_world.physics_world)
    for i in range(1, 100000 if render else 2000):
        o, r, d, info = env.step([0, 1])
        if render:
            env.render()
        if d:
            assert info["crash"], "Can not crash with object!"
            break
    env.close()


if __name__ == "__main__":
    test_object_collision(render=True)
