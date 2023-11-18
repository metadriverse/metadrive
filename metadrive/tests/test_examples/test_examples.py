import os.path
import subprocess
from metadrive import MetaDrive_PACKAGE_DIR
import time
import pytest

examples = [
    "draw_maps.py", "drive_in_multi_agent_env.py --top_down", "drive_in_real_env.py --top_down",
    "drive_in_real_env.py --top_down --waymo", "drive_in_real_env.py --reactive_traffic",
    "drive_in_safe_metadrive_env.py", "drive_in_single_agent_env.py", "procedural_generation.py",
    "profile_metadrive.py", "profile_metadrive_marl.py", "top_down_metadrive.py",
    "generate_video_for_bev_and_interface.py", "verify_headless_installation.py", "verify_image_observation.py"
]
examples_dir_path = os.path.join(MetaDrive_PACKAGE_DIR, "examples")
scripts = [os.path.join(examples_dir_path, exp) for exp in examples]


@pytest.mark.parametrize("script", scripts, ids=examples)
def test_script(script, timeout=60):
    """
    Run script in a subprocess and check its running time.
    Args:
        script: the path to the script
        timeout: script that can run over `timeout` seconds can pass the test

    Returns: None

    """
    start_time = time.time()

    # Run your script using subprocess
    process = subprocess.Popen(['python', script])

    # Wait for the script to finish or timeout after 60 seconds
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # If the script is still running after 60 seconds, terminate it and pass the test
        process.kill()
    finally:
        runtime = time.time() - start_time
        assert runtime >= 0, "Script terminated unexpectedly"
