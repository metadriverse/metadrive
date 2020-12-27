from pgdrive.examples.ppo_expert.numpy_expert import expert


def get_terminal_state(info):
    if info["crash"]:
        state = "Crash"
    elif info["out_of_road"]:
        state = "Out of Road"
    elif info["arrive_dest"]:
        state = "Success"
    else:
        state = "Max Step"
    return state
