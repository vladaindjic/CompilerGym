# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Fuzz test for LlvmEnv.validate()."""
import numpy as np
import pytest

from compiler_gym.datasets import BenchmarkInitError
from compiler_gym.envs import LlvmEnv
from tests.pytest_plugins.random_util import apply_random_trajectory
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


# The uniform range for trajectory lengths.
RANDOM_TRAJECTORY_LENGTH_RANGE = (1, 50)


@pytest.mark.timeout(600)
def test_fuzz(env: LlvmEnv, reward_space: str):
    """This test produces a random trajectory, resets the environment, then
    replays the trajectory and checks that it produces the same state.
    """
    env.observation_space = "Autophase"
    env.reward_space = reward_space
    benchmark = env.datasets["generator://csmith-v0"].random_benchmark()
    print(benchmark.uri)  # For debugging in case of failure.

    try:
        env.reset(benchmark=benchmark)
    except BenchmarkInitError:
        return

    trajectory = apply_random_trajectory(
        env, random_trajectory_length_range=RANDOM_TRAJECTORY_LENGTH_RANGE
    )
    print(env.state)  # For debugging in case of failure.
    env.reset(benchmark=benchmark)

    for i, (action, observation, reward, done) in enumerate(trajectory, start=1):
        print(f"Replaying step {i}: {env.action_space.flags[action]}")
        replay_observation, replay_reward, replay_done, info = env.step(action)
        assert done == replay_done, info

        np.testing.assert_array_almost_equal(observation, replay_observation)
        np.testing.assert_almost_equal(reward, replay_reward)


if __name__ == "__main__":
    main()
