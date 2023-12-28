from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.tests.test_functionality.test_memory_leak_engine import process_memory
import matplotlib.pyplot as plt


def local_test_close_and_restart(repeat=100):
    """
    Test the memory leak
    """
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    memory_usage = []
    try:
        for m in ["X", "O", "C", "S", "R", "r", "T"] * repeat:
            env = MetaDriveEnv({"map": m, "use_render": False})
            o, _ = env.reset()
            env.close()
            memory = process_memory(to_mb=True)
            memory_usage.append(memory)
            ax.clear()
            ax.plot(memory_usage)
            ax.set_xlabel('Step')
            ax.set_ylabel('Memory Usage')
            plt.pause(0.05)  # A short pause to update the plot
    finally:
        if "env" in locals():
            env = locals()["env"]
            env.close()

        plt.ioff()  # Turn off interactive mode
        plt.show()


if __name__ == '__main__':
    local_test_close_and_restart(100)
