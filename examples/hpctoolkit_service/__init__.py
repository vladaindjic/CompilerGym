"""This module defines and registers the example gym environments."""
import subprocess
from pathlib import Path
from typing import Iterable

from compiler_gym.datasets import Benchmark, Dataset
from compiler_gym.spaces import Reward
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path, site_data_path

HPCTOOLKIT_PY_SERVICE_BINARY: Path = runfiles_path(
    "examples/hpctoolkit_service/service_py/compiler_gym-example-service-py"
)

BENCHMARKS_PATH: Path = runfiles_path(
    "examples/hpctoolkit_service/benchmarks/cpu-benchmarks"
)


HPCTOOLKIT_HEADER: Path = runfiles_path(
    "/home/dx4/tools/CompilerGym/compiler_gym/third_party/hpctoolkit/header.h"
)

import pdb


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
        #     self.baseline_runtime = 0

        # def reset(self, benchmark: str, observation_view):
        #     del benchmark  # unused
        #     self.baseline_runtime = observation_view["runtime"]

        # def update(self, action, observations, observation_view):
        #     del action
        #     del observation_view
        #     return float(self.baseline_runtime - observations[0]) / self.baseline_runtime
        self.previous_runtime = None

    def reset(self, benchmark: str, observation_view):
        del benchmark  # unused
        self.previous_runtime = None

    def update(self, action, observations, observation_view):
        del action
        del observation_view
        pdb.set_trace()

        if self.previous_runtime is None:
            self.previous_runtime = observations[0]
        reward = float(self.previous_runtime - observations[0])
        self.previous_runtime = observations[0]
        return reward


class HPCToolkitReward(Reward):
    """An example reward that uses changes in the "runtime" observation value
    to compute incremental reward.
    """

    def __init__(self):
        super().__init__(
            id="hpctoolkit",
            observation_spaces=["hpctoolkit"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.previous_runtime = None

    def reset(self, benchmark: str, observation_view):
        del benchmark  # unused
        self.previous_runtime = None

    def update(self, action, observations, observation_view):
        del action
        del observation_view
        pdb.set_trace()

        if self.previous_runtime is None:
            self.previous_runtime = observations[0]
        reward = float(self.previous_runtime - observations[0])
        self.previous_runtime = observations[0]
        return reward


class HPCToolkitDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="benchmark://hpctoolkit-cpu-v0",
            license="MIT",
            description="HPCToolkit cpu dataset",
            site_data_base=site_data_path("example_dataset"),
        )
        self._benchmarks = {
            "benchmark://hpctoolkit-cpu-v0/offsets1": Benchmark.from_file_contents(
                "benchmark://hpctoolkit-cpu-v0/offsets1",
                self.preprocess(BENCHMARKS_PATH / "offsets1.c"),
            ),
            "benchmark://hpctoolkit-cpu-v0/conv2d": Benchmark.from_file_contents(
                "benchmark://hpctoolkit-cpu-v0/conv2d",
                self.preprocess(BENCHMARKS_PATH / "conv2d.c"),
            ),
        }

    def benchmark_uris(self) -> Iterable[str]:
        yield from self._benchmarks.keys()

    def benchmark(self, uri: str) -> Benchmark:
        if uri in self._benchmarks:
            return self._benchmarks[uri]
        else:
            raise LookupError("Unknown program name")


register(
    id="hpctoolkit-llvm",
    entry_point="compiler_gym.envs:CompilerEnv",
    kwargs={
        "service": HPCTOOLKIT_PY_SERVICE_BINARY,
        "rewards": [RuntimeReward(), HPCToolkitReward()],
        "datasets": [HPCToolkitDataset()],
    },
)
