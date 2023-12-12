import importlib
if importlib.util.find_spec("torch") is not None:
    from metadrive.examples.ppo_expert.torch_expert import torch_expert as expert
else:
    from metadrive.examples.ppo_expert.numpy_expert import expert
