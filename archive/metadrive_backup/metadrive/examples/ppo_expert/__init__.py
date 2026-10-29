try:
    import torch

    assert hasattr(torch, "device")
    from metadrive.examples.ppo_expert.torch_expert import torch_expert as expert
except:
    from metadrive.examples.ppo_expert.numpy_expert import expert
