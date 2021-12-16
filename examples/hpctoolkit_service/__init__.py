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

BENCHMARKS_PATH: Path = runfiles_path("examples/hpctoolkit_service/benchmarks")

import pdb


class RuntimeReward(Reward):
    """An example reward that uses changes in the "runtime" observation value
    to compute incremental reward.
    """

    def __init__(self):
        super().__init__(
            id="hatchet",
            observation_spaces=["hatchet"],
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


class HPCToolkitDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="benchmark://gpukernels-v0",
            license="MIT",
            description="An example of gpu kernel dataset",
            site_data_base=site_data_path("example_dataset"),
        )
        self._benchmarks = {
            # "benchmark://gpukernels-v0/cuda-driver": Benchmark.from_file_contents(
            #     "benchmark://gpukernels-v0/cuda-driver",
            #     self.preprocess(BENCHMARKS_PATH / "vecSet1.cu"),
            # ),
            "benchmark://gpukernels-v0/cuda-runtime": Benchmark.from_file(
                "benchmark://gpukernels-v0/cuda-runtime",
                BENCHMARKS_PATH / "cuda-runtime/main.cu",
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
    id="hpctoolkit-gpukernels",
    entry_point="compiler_gym.envs:CompilerEnv",
    kwargs={
        "service": HPCTOOLKIT_PY_SERVICE_BINARY,
        "rewards": [RuntimeReward()],
        "datasets": [HPCToolkitDataset()],
    },
)
