########################
Training with RLLib
########################


We provide a script demonstrating how to use `RLLib <https://docs.ray.io/en/latest/rllib.html>`_ to
train RL agents:

.. code-block:: shell

    # Make sure current folder does not have a sub-folder named metadrive
    python -m metadrive.examples.train_generalization_experiment

    # You can also use GPUs and customized experiment name:
    python -m metadrive.examples.train_generalization_experiment \
     --exp-name CUSTOMIZED_EXP_NAME \
     --num-gpus HOW_MANY_GPUS_IN_THIS_MACHINES


In this example, we leave the training hyper-parameter :code:`config["num_envs_per_worker"] = 1` as default, so that each process (ray worker) will only contain one MetaDrive instance.
We further set the evaluation workers :code:`config["evaluation_num_workers"] = 5`, so that the test set environments are hosted in separated processes.
By utilizing the feature of RLLib, we avoid the issue of multiple MetaDrive instances in single process.

We welcome more examples using MetaDrive in different context! Please show off your code if you like to share it by opening new issue! Thanks!

.. note:: We tested this script using :code:`ray==1.2.0`. If you find this script not compatible with newer RLLib, please contact us.