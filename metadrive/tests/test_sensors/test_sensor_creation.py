from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.envs.metadrive_env import MetaDriveEnv


def test_creation():
    """
    Test sensor creation.
    """

    # image_observation: True, use_render: True, request main_camera
    env = MetaDriveEnv(
        {
            "show_terrain": False,
            "window_size": (16, 16),
            "sensors": {
                "semantic": (SemanticCamera, 400, 300),
                "rgb": (RGBCamera, 400, 300),
                "main_camera": (),
            },
            "vehicle_config": dict(image_source="main_camera"),
            "interface_panel": ["rgb", "main_camera", "dashboard"],
            "image_observation": True,
            "use_render": True,
        }
    )
    env.reset()
    assert env.config["sensors"].keys() == {
        "lidar", "side_detector", "lane_line_detector", "semantic", "rgb", "main_camera", "dashboard"
    } == env.engine.sensors.keys()
    assert env.config["interface_panel"] == ["rgb", "dashboard"]
    assert not env.engine.main_window_disabled
    env.close()

    # image_observation: True, use_render: False, request main_camera, dashboard
    env = MetaDriveEnv(
        {
            "show_terrain": False,
            "window_size": (16, 16),
            "sensors": {
                "semantic": (SemanticCamera, 400, 300),
                "rgb": (RGBCamera, 400, 300),
                "main_camera": (),
            },
            "vehicle_config": dict(image_source="main_camera"),
            "interface_panel": ["rgb", "dashboard"],
            "image_observation": True,
            "use_render": False,
        }
    )
    env.reset()
    assert env.config["sensors"].keys() == {
        "lidar", "side_detector", "lane_line_detector", "semantic", "rgb", "dashboard", "main_camera"
    } == env.engine.sensors.keys()
    assert env.config["interface_panel"] == ["rgb", "dashboard"]
    assert not env.engine.main_window_disabled
    env.close()

    # image_observation: False
    env = MetaDriveEnv(
        {
            "show_terrain": False,
            "window_size": (16, 16),
            "sensors": {
                "semantic": (SemanticCamera, 400, 300),
                "rgb": (RGBCamera, 400, 300),
            },
            "interface_panel": ["rgb", "dashboard"],
            "image_observation": False,  # it is a switch telling metadrive to use rgb as observation
        }
    )
    env.reset()
    assert env.config["sensors"].keys() == {"lidar", "side_detector", "lane_line_detector"} == env.engine.sensors.keys()
    assert env.config["interface_panel"] == []
    assert env.engine.main_window_disabled
    env.close()

    # image_observation: False, request main_camera
    env = MetaDriveEnv(
        {
            "show_terrain": False,
            "window_size": (16, 16),
            "sensors": {
                "semantic": (SemanticCamera, 400, 300),
                "rgb": (RGBCamera, 400, 300),
                "main_camera": (),
            },
            "interface_panel": ["rgb", "dashboard"],
            "image_observation": False,  # it is a switch telling metadrive to use rgb as observation
        }
    )
    env.reset()
    assert env.config["sensors"].keys() == {"lidar", "side_detector", "lane_line_detector"} == env.engine.sensors.keys()
    assert env.config["interface_panel"] == []
    assert env.engine.main_window_disabled
    env.close()

    # image_observation: False, use_render: True
    env = MetaDriveEnv(
        {
            "show_terrain": False,
            "window_size": (16, 16),
            "sensors": {
                "semantic": (SemanticCamera, 400, 300),
                "rgb": (RGBCamera, 400, 300),
            },
            "interface_panel": ["rgb", "dashboard"],
            "image_observation": False,
            "use_render": True,
        }
    )
    env.reset()
    assert env.config["sensors"].keys() == {
        "lidar", "side_detector", "lane_line_detector", "semantic", "rgb", "main_camera", "dashboard"
    } == env.engine.sensors.keys()
    assert env.config["interface_panel"] == ["rgb", "dashboard"]
    assert not env.engine.main_window_disabled
    env.close()

    # image_observation: True, use_render: False
    env = MetaDriveEnv(
        {
            "show_terrain": False,
            "window_size": (16, 16),
            "sensors": {
                "semantic": (SemanticCamera, 400, 300),
                "rgb": (RGBCamera, 400, 300),
            },
            "vehicle_config": dict(image_source="rgb"),
            "interface_panel": ["rgb", "dashboard"],
            "image_observation": True,
            "use_render": False,
        }
    )
    env.reset()
    assert env.config["sensors"].keys() == {"lidar", "side_detector", "lane_line_detector", "semantic", "rgb"
                                            } == env.engine.sensors.keys()
    assert env.config["interface_panel"] == []
    assert env.engine.main_window_disabled
    env.close()

    # image_observation: True, use_render: True
    env = MetaDriveEnv(
        {
            "show_terrain": False,
            "window_size": (16, 16),
            "sensors": {
                "semantic": (SemanticCamera, 400, 300),
                "rgb": (RGBCamera, 400, 300),
            },
            "vehicle_config": dict(image_source="rgb"),
            "interface_panel": ["rgb", "dashboard"],
            "image_observation": True,
            "use_render": True,
        }
    )
    env.reset()
    assert env.config["sensors"].keys() == {
        "lidar", "side_detector", "lane_line_detector", "semantic", "rgb", "dashboard", "main_camera"
    } == env.engine.sensors.keys()
    assert env.config["interface_panel"] == ["rgb", "dashboard"]
    assert not env.engine.main_window_disabled
    env.close()

    # image_observation: True, use_render: False, request main_camera
    env = MetaDriveEnv(
        {
            "show_terrain": False,
            "window_size": (16, 16),
            "sensors": {
                "semantic": (SemanticCamera, 400, 300),
                "rgb": (RGBCamera, 400, 300),
                "main_camera": (),
            },
            "vehicle_config": dict(image_source="rgb"),
            "interface_panel": ["rgb", "dashboard"],
            "image_observation": True,
            "use_render": False,
        }
    )
    env.reset()
    assert env.config["sensors"].keys() == {
        "lidar", "side_detector", "lane_line_detector", "semantic", "rgb", "dashboard", "main_camera"
    } == env.engine.sensors.keys()
    assert env.config["interface_panel"] == ["rgb", "dashboard"]
    assert not env.engine.main_window_disabled
    env.close()

    # image_observation: False, use_render: True, request main_camera
    env = MetaDriveEnv(
        {
            "show_terrain": False,
            "window_size": (16, 16),
            "sensors": {
                "semantic": (SemanticCamera, 400, 300),
                "rgb": (RGBCamera, 400, 300),
                "main_camera": (),
            },
            "vehicle_config": dict(image_source="main_camera"),
            "interface_panel": ["rgb", "main_camera"],
            "image_observation": False,
            "use_render": True,
        }
    )
    env.reset()
    assert env.config["sensors"].keys() == {
        "lidar", "side_detector", "lane_line_detector", "semantic", "rgb", "main_camera"
    } == env.engine.sensors.keys()
    assert env.config["interface_panel"] == ["rgb"]
    assert not env.engine.main_window_disabled
    env.close()

    # image_observation: False, use_render: False, request main_camera
    env = MetaDriveEnv(
        {
            "show_terrain": False,
            "window_size": (16, 16),
            "sensors": {
                "semantic": (SemanticCamera, 400, 300),
                "rgb": (RGBCamera, 400, 300),
                "main_camera": (),
            },
            "vehicle_config": dict(image_source="main_camera"),
            "interface_panel": ["rgb", "main_camera"],
            "image_observation": False,
            "use_render": False,
        }
    )
    env.reset()
    assert env.config["sensors"].keys() == {
        "lidar",
        "side_detector",
        "lane_line_detector",
    } == env.engine.sensors.keys()
    assert env.config["interface_panel"] == []
    assert env.engine.main_window_disabled
    env.close()


if __name__ == '__main__':
    test_creation()
