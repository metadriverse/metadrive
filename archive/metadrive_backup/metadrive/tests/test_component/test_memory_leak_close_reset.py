from metadrive.envs import ScenarioEnv, MetaDriveEnv
from metadrive.tests.test_functionality.test_memory_leak_engine import process_memory


def test_close_and_restart():
    """
    Test the memory leak
    """
    start = 11
    start_memory = None
    try:
        for m in range(50):
            env = ScenarioEnv({"use_render": False, "log_level": 50})
            o, _ = env.reset()
            env.close()
            memory = process_memory(to_mb=True)
            if m == start:
                start_memory = memory
        assert (memory - start_memory) / (m - start) < 0.4, "Memory leak per close-reset should be less than 0.4 mb"
    finally:
        if "env" in locals():
            env = locals()["env"]
            env.close()


def test_close_and_restart_metadrive_env():
    """
    Test the memory leak
    """
    start = 11
    start_memory = None
    try:
        for idx, m in enumerate([
                "X",
                "O",
                "C",
                "R",
                "r",
        ] * 20):
            env = MetaDriveEnv({"map": m, "use_render": False, "log_level": 50, "traffic_density": 0})
            o, _ = env.reset()
            env.close()
            memory = process_memory(to_mb=True)
            if idx == start:
                start_memory = memory
        assert (memory - start_memory) / (idx - start) < 0.4, "Memory leak per close-reset should be less than 0.4 mb"
    finally:
        if "env" in locals():
            env = locals()["env"]
            env.close()
