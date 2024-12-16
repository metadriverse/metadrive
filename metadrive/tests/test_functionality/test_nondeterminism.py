"""
Credit:  https://github.com/olek-osikowicz/metadrive-nondeterminism/blob/master/reproduce_nondeterminism_bug.ipynb
Issue:  https://github.com/metadriverse/metadrive/issues/758

Usage: run this file. or pytest.
"""
import copy
import os

import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from metadrive import MetaDriveEnv
from metadrive.constants import TerminationState
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.examples.ppo_expert.numpy_expert import expert


def assert_dict_almost_equal(dict1, dict2, tol=1e-6):
    """
    Recursively assert that two dictionaries are almost equal.
    Allows for tiny differences (less than tol) using numpy's allclose function.
    """
    assert dict1.keys() == dict2.keys(), f"Keys mismatch: {dict1.keys()} != {dict2.keys()}"

    for key in dict1:
        val1 = dict1[key]
        val2 = dict2[key]

        if isinstance(val1, dict) and isinstance(val2, dict):
            assert_dict_almost_equal(val1, val2, tol)
        else:

            if isinstance(val1, str):
                assert val1 == val2, f"Values for key '{key}' are not equal: {val1} != {val2}"
            elif np.isscalar(val1) and np.isnan(val1):
                assert np.isnan(val2).all(), f"Values for key '{key}' are not equal: {val1} != {val2}"
            else:
                assert np.allclose(
                    val1, val2, rtol=tol
                ), f"Values for key '{key}' are not almost equal: {val1} != {val2}"


# Example usage:


def are_traces_deterministic(traces) -> bool:
    df = pd.DataFrame(traces)

    # grouping by repetition to get a list of traces
    traces = df.groupby("repetition")

    # drop index and repetition ID to compare only step info later
    stripped_traces = [trace.reset_index(drop=True).drop("repetition", axis=1) for _, trace in traces]

    # iterate over each trace and check if it is equal to the first one
    are_equal_to_first_trace = [trace.equals(stripped_traces[0]) for trace in stripped_traces]

    first_trace = stripped_traces[0].to_dict()
    for trace in stripped_traces:
        # Assert
        trace = trace.to_dict()
        assert_dict_almost_equal(first_trace, trace)


@pytest.mark.parametrize(
    "n_scenarios, seed, expert_driving, force_step", [
        (10, 0, True, 0),
        (10, 0, False, 0),
        (10, 1, True, 0),
        (10, 1, False, 0),
        (10, 2, True, 1),
        (10, 3, False, 1),
        (10, 3, False, 10),
        (10, 3, True, 10),
        (10, 3, True, 50),
        (10, 3, False, 50),
    ]
)
def test_determinism_reset(n_scenarios, seed, expert_driving, force_step) -> list:
    """
    Runs same scenario n time and collects the traces
    """

    traces = []
    try:
        env = MetaDriveEnv(config={"map": "C", "num_scenarios": n_scenarios})

        for rep in range(n_scenarios):

            obs, step_info = env.reset(seed)
            step_info["repetition"] = rep
            traces.append(step_info)
            print(f"{env.current_seed = }")
            step = 0

            while True:
                # get action from expert driving, or a dummy action

                if step >= force_step:
                    break

                if expert_driving:
                    action, exp_obs = expert(env.agent, deterministic=True, need_obs=True)
                    print(1111)
                else:
                    action = [0, 0.33]

                obs, reward, tm, tr, step_info = env.step(action)
                step_info["repetition"] = rep
                traces.append(step_info)

                step += 1

                if tm or tr:
                    break

    finally:
        env.close()

    are_traces_deterministic(traces)
    return traces


@pytest.mark.parametrize(
    "n_scenarios, seed, expert_driving, force_step", [
        (10, 0, True, 0),
        (10, 1, False, 0),
        (10, 2, True, 1),
        (10, 3, False, 1),
        (10, 3, True, 10),
        (10, 3, False, 10),
        (10, 3, True, 50),
        (10, 3, False, 50),
    ]
)
def test_determinism_close(n_scenarios, seed, expert_driving, force_step) -> list:
    """
    Runs same scenario n time and collects the traces
    """

    traces = []
    try:

        for rep in range(n_scenarios):
            env = MetaDriveEnv(config={"map": "C", "num_scenarios": n_scenarios})

            obs, step_info = env.reset(seed)
            step_info["repetition"] = rep
            traces.append(step_info)
            print(f"{env.current_seed = }")
            step = 0
            while True:

                if step >= force_step:
                    break

                # get action from expert driving, or a dummy action
                action = (expert(env.agent, deterministic=True) if expert_driving else [0, 0.33])
                obs, reward, tm, tr, step_info = env.step(action)
                step_info["repetition"] = rep
                traces.append(step_info)

                step += 1

                if tm or tr:
                    break

            env.close()
    finally:
        pass

    are_traces_deterministic(traces)
    return traces


if __name__ == '__main__':
    test_determinism_reset(10, 2, True, 5)
