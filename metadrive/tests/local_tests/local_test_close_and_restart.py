from metadrive.envs.base_env import BaseEnv
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.tests.test_functionality.test_memory_leak_engine import process_memory
import matplotlib.pyplot as plt
import tracemalloc

tracemalloc.start()


def local_test_close_and_restart(repeat=100):
    """
    Test the memory leak
    """
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    memory_usage = []
    try:
        for m in ["X", "O", "C", "R", "r", ] * 40:
            # env = MetaDriveEnv({"map": m, "use_render": False})
            env = BaseEnv({"use_render": False, "log_level": 50})
            o, _ = env.reset()
            env.close()
            memory = process_memory(to_mb=True)
            print(process_memory(to_mb=True))
            memory_usage.append(memory)
            ax.clear()
            ax.plot(memory_usage)
            ax.set_xlabel('Step')
            ax.set_ylabel('Memory Usage')
            plt.pause(0.05)  # A short pause to update the plot

            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            print("[ Top 10 ]")
            for stat in top_stats[:10]:
                print(stat)

    finally:
        if "env" in locals():
            env = locals()["env"]
            env.close()

        plt.ioff()  # Turn off interactive mode
        plt.show()


if __name__ == '__main__':
    local_test_close_and_restart(100)
