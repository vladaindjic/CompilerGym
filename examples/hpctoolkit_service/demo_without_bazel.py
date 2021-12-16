# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This script demonstrates how the Python example service without needing
to use the bazel build system. Usage:

    $ python hpctoolkit_service/demo_without_bazel.py

It is equivalent in behavior to the demo.py script in this directory.
"""
import logging
import os
import pdb
import subprocess
from pathlib import Path
from typing import Iterable

# import pdb
import gym

from compiler_gym.datasets import Benchmark, Dataset
from compiler_gym.spaces import Reward
from compiler_gym.util.logging import init_logging
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import site_data_path

EXAMPLE_PY_SERVICE_BINARY: Path = Path(
    "hpctoolkit_service/service_py/example_service.py"
)
assert EXAMPLE_PY_SERVICE_BINARY.is_file(), "Service script not found"

BENCHMARKS_PATH: Path = "/home/dx4/tools/CompilerGym/examples/hpctoolkit_service/benchmarks/cuda-runtime/main.cu"


class RuntimeReward(Reward):
    """An example reward that uses changes in the "runtime" observation value
    to compute incremental reward.
    """

    def __init__(self):
        super().__init__(
            id="runtime",
            observation_spaces=["runtime"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.baseline_runtime = 0

    def reset(self, benchmark: str, observation_view):
        del benchmark  # unused
        self.baseline_runtime = observation_view["runtime"]

    def update(self, action, observations, observation_view):
        del action
        del observation_view
        return float(self.baseline_runtime - observations[0]) / self.baseline_runtime


class ExampleDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="benchmark://gpukernels-v0",
            license="MIT",
            description="An example dataset",
            site_data_base=site_data_path("example_dataset"),
        )
        self._benchmarks = {
            "benchmark://gpukernels-v0/cuda-runtime": Benchmark.from_file(
                "benchmark://gpukernels-v0/cuda-runtime",
                BENCHMARKS_PATH,
            ),
        }

    def benchmark_uris(self) -> Iterable[str]:
        yield from self._benchmarks.keys()

    def benchmark(self, uri: str) -> Benchmark:
        if uri in self._benchmarks:
            return self._benchmarks[uri]
        else:
            raise LookupError("Unknown program name")


# Register the environment for use with gym.make(...).
register(
    id="gpukernels-v0",
    entry_point="compiler_gym.envs:CompilerEnv",
    kwargs={
        "service": EXAMPLE_PY_SERVICE_BINARY,
        "rewards": [RuntimeReward()],
        "datasets": [ExampleDataset()],
    },
)


def main():
    # Use debug verbosity to print out extra logging information.
    init_logging(level=logging.DEBUG)

    # Create the environment using the regular gym.make(...) interface.
    with gym.make("gpukernels-v0") as env:
        env.reset()
        for _ in range(2):
            observation, reward, done, info = env.step(
                action=env.action_space.sample(),
                observations=["hatchet"],
                rewards=["runtime"],
            )
            print(reward)
            print(observation)
            print(info)
            # #pdb.set_trace()
            if done:
                env.reset()


if __name__ == "__main__":
    main()
