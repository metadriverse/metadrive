import tracemalloc
import matplotlib.pyplot as plt

from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.tests.test_functionality.test_memory_leak_engine import process_memory

# tracemalloc.start()


def local_test_close_and_restart(repeat=100):
    """
    Test the memory leak
    """
    draw = False
    if draw:
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        memory_usage = []
    snapshot = None
    try:
        for m in [
                "X",
                "O",
                "C",
                "R",
                "r",
        ] * 40:
            # env = MetaDriveEnv({"map": m, "use_render": False, "log_level": 50, "traffic_density": 0})
            # env = BaseEnv({"use_render": False, "log_level": 50})
            env = ScenarioEnv({"use_render": False, "log_level": 50})
            o, _ = env.reset()
            env.close()
            memory = process_memory(to_mb=True)
            print(memory)
            if draw:
                memory_usage.append(memory)
                ax.clear()
                ax.plot(memory_usage)
                ax.set_xlabel('Step')
                ax.set_ylabel('Memory Usage')
                plt.pause(0.05)  # A short pause to update the plot

            # new_snapshot = tracemalloc.take_snapshot()
            # if snapshot:
            #     stats = new_snapshot.compare_to(snapshot, 'lineno')
            #     print("[ Top 10 ]")
            #     for stat in stats[:10]:
            #         print(stat)
            # snapshot = new_snapshot

    finally:
        if "env" in locals():
            env = locals()["env"]
            env.close()

        plt.ioff()  # Turn off interactive mode
        plt.show()


if __name__ == '__main__':
    local_test_close_and_restart(100)
