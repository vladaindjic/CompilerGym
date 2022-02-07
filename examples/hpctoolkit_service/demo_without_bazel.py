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
import pickle
import subprocess
from pathlib import Path
from typing import Iterable

import gym
import hatchet as ht

from compiler_gym.datasets import Benchmark, Dataset, BenchmarkUri
from compiler_gym.envs.llvm.datasets import CBenchDataset, CBenchLegacyDataset2, CBenchLegacyDataset, CsmithDataset, \
    NPBDataset, BlasDataset, AnghaBenchDataset, CHStoneDataset
from compiler_gym.envs.llvm.llvm_benchmark import get_system_includes
from compiler_gym.spaces import Reward
from compiler_gym.third_party import llvm
from compiler_gym.util.logging import init_logging
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path, site_data_path

reward_metric = "REALTIME (sec) (I)"  # "time (inc)"


HPCTOOLKIT_PY_SERVICE_BINARY: Path = Path(
    "hpctoolkit_service/service_py/example_service.py"
)
assert HPCTOOLKIT_PY_SERVICE_BINARY.is_file(), "Service script not found"

# BENCHMARKS_PATH: Path = runfiles_path("examples/hpctoolkit_service/benchmarks")
BENCHMARKS_PATH: Path = (
    "/home/vi3/CompilerGym/examples/hpctoolkit_service/benchmarks/cpu-benchmarks"
)

HPCTOOLKIT_HEADER: Path = Path(
    "/home/vi3/CompilerGym/compiler_gym/third_party/hpctoolkit/header.h"
)


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
        print("Reward Runtime: reset")
        del benchmark  # unused
        self.baseline_runtime = observation_view["runtime"]

    def update(self, action, observations, observation_view):
        print("Reward Runtime: update")
        del action
        del observation_view
        return float(self.baseline_runtime - observations[0]) / self.baseline_runtime


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
        self.baseline_cct = None
        self.baseline_runtime = 0

    def reset(self, benchmark: str, observation_view):
        print("Reward HPCToolkit: reset")
        # pdb.set_trace()
        del benchmark  # unused
        unpickled_cct = observation_view["hpctoolkit"]
        gf = pickle.loads(unpickled_cct)
        self.baseline_cct = gf
        self.baseline_runtime = gf.dataframe[reward_metric][0]

    def update(self, action, observations, observation_view):
        print("Reward HPCToolkit: update")
        # pdb.set_trace()
        del action
        del observation_view

        gf = pickle.loads(observations[0])
        new_runtime = gf.dataframe[reward_metric][0]
        return float(self.baseline_runtime - new_runtime) / self.baseline_runtime


class HPCToolkitDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="benchmark://hpctoolkit-cpu-v0",
            license="MIT",
            description="HPCToolkit cpu dataset",
            site_data_base=site_data_path("example_dataset"),
        )

        # self._benchmarks = {
        #     "benchmark://hpctoolkit-cpu-v0/offsets1": Benchmark.from_file(
        #         "benchmark://hpctoolkit-cpu-v0/offsets1",
        #         BENCHMARKS_PATH + "/offsets1.c",
        #     ),
        #     "benchmark://hpctoolkit-cpu-v0/conv2d": Benchmark.from_file(
        #         "benchmark://hpctoolkit-cpu-v0/conv2d",
        #         BENCHMARKS_PATH + "/conv2d.c",
        #     ),
        # }

        self._benchmarks = {
            "benchmark://hpctoolkit-cpu-v0/conv2d": Benchmark.from_file_contents(
                "benchmark://hpctoolkit-cpu-v0/conv2d",
                self.preprocess(BENCHMARKS_PATH + "/conv2d.c"),
            ),
            "benchmark://hpctoolkit-cpu-v0/offsets1": Benchmark.from_file_contents(
                "benchmark://hpctoolkit-cpu-v0/offsets1",
                self.preprocess(BENCHMARKS_PATH + "/offsets1.c"),
            ),
            "benchmark://hpctoolkit-cpu-v0/nanosleep": Benchmark.from_file_contents(
                "benchmark://hpctoolkit-cpu-v0/nanosleep",
                self.preprocess(BENCHMARKS_PATH + "/nanosleep.c"),
            ),
        }

    @staticmethod
    def preprocess(src: Path) -> bytes:
        """Front a C source through the compiler frontend."""
        # TODO(github.com/facebookresearch/CompilerGym/issues/325): We can skip
        # this pre-processing, or do it on the service side, once support for
        # multi-file benchmarks lands.
        cmd = [
            str(llvm.clang_path()),
            "-E",
            "-o",
            "-",
            "-I",
            str(HPCTOOLKIT_HEADER.parent),
            src,
        ]
        for directory in get_system_includes():
            cmd += ["-isystem", str(directory)]
        return subprocess.check_output(
            cmd,
            timeout=300,
        )

    def benchmark_uris(self) -> Iterable[str]:
        yield from self._benchmarks.keys()

    def benchmark(self, uri: str) -> Benchmark:
        if uri in self._benchmarks:
            return self._benchmarks[uri]
        else:
            raise LookupError("Unknown program name")

    def benchmark_from_parsed_uri(self, uri: BenchmarkUri) -> Benchmark:
        # TODO: IMPORTANT
        return self.benchmark(str(uri))

# Register the environment for use with gym.make(...).
register(
    id="hpctoolkit-llvm-v0",
    entry_point="compiler_gym.envs:CompilerEnv",
    kwargs={
        "service": HPCTOOLKIT_PY_SERVICE_BINARY,
        # "rewards": [RuntimeReward(), HPCToolkitReward()],
        "rewards": [RuntimeReward()],
        "datasets": [HPCToolkitDataset(), CBenchDataset(site_data_path("llvm-v0")),
                     CsmithDataset(site_data_path("llvm-v0"), sort_order=0),
                     NPBDataset(site_data_path("llvm-v0"), sort_order=0),
                     BlasDataset(site_data_path("llvm-v0"), sort_order=0),
                     AnghaBenchDataset(site_data_path("llvm-v0"), sort_order=0),
                     CHStoneDataset(site_data_path("llvm-v0"), sort_order=0),
                     ],
        # "datasets": [HPCToolkitDataset()],
    },
)


def main():
    # Use debug verbosity to print out extra logging information.
    init_logging(level=logging.DEBUG)

    # Create the environment using the regular gym.make(...) interface.
    with gym.make("hpctoolkit-llvm-v0") as env:

        # env.reset(benchmark="benchmark://cbench-v1/qsort")

        benchmark_to_process = [
            # from benchmarks directory
            # "benchmark://hpctoolkit-cpu-v0/offsets1",
            # "benchmark://hpctoolkit-cpu-v0/conv2d",
            # "benchmark://hpctoolkit-cpu-v0/nanosleep",

            # cbench-v1
            "benchmark://cbench-v1/bitcount",
            "benchmark://cbench-v1/qsort",
            "benchmark://cbench-v1/blowfish",
            "benchmark://cbench-v1/bzip2",
            "benchmark://cbench-v1/crc32",
            "benchmark://cbench-v1/dijkstra",
            # "benchmark://cbench-v1/gsm",                # FIXME: ValueError: 'utf-8' codec can't decode byte 0xcb in position 2: invalid continuation byte
            "benchmark://cbench-v1/jpeg-c",
            "benchmark://cbench-v1/jpeg-d",
            "benchmark://cbench-v1/patricia",
            "benchmark://cbench-v1/sha",
            "benchmark://cbench-v1/stringsearch",
            "benchmark://cbench-v1/susan",
            "benchmark://cbench-v1/tiff2bw",
            "benchmark://cbench-v1/tiff2rgba",
            "benchmark://cbench-v1/tiffdither",
            "benchmark://cbench-v1/tiffmedian",

            # csmith
            "generator://csmith-v0/0",
            "generator://csmith-v0/1",
            "generator://csmith-v0/2",
            # ...
            # The number represents the seed which needs to be less than or equal to UINT_MAX = (2 ** 32) - 1
            "generator://csmith-v0/23",
            "generator://csmith-v0/33",
            "generator://csmith-v0/1123",

            # ===========================
            # npb
            # "benchmark://npb-v0/3"
            # warning: overriding the module target triple with x86_64-unknown-linux-gnu [-Woverride-module]
            # 1 warning generated.
            # /usr/lib/gcc/x86_64-redhat-linux/8/../../../../lib64/crt1.o: In function `_start':
            # (.text+0x24): undefined reference to `main'
            # clang-12: error: linker command failed with exit code 1 (use -v to see invocation
            # ====================================
            #

            # ====================================
            # "benchmark://blas-v0/1",
            # blas
            # /usr/lib/gcc/x86_64-redhat-linux/8/../../../../lib64/crt1.o: In function `_start':
            # (.text+0x24): undefined reference to `main'
            # /tmp/benchmark-21b6f1.o: In function `dtbsv_':
            # /home/shoshijak/Documents/blas_ir/BLAS-3.8.0/dtbsv.f:230: undefined reference to `lsame_'
            # /home/shoshijak/Documents/blas_ir/BLAS-3.8.0/dtbsv.f:230: undefined reference to `lsame_'
            # /home/shoshijak/Documents/blas_ir/BLAS-3.8.0/dtbsv.f:232: undefined reference to `lsame_'
            # /home/shoshijak/Documents/blas_ir/BLAS-3.8.0/dtbsv.f:232: undefined reference to `lsame_'
            # /home/shoshijak/Documents/blas_ir/BLAS-3.8.0/dtbsv.f:232: undefined reference to `lsame_'
            # ====================================

            # ====================================
            # Maybe we could access to the .c code directly.
            # "benchmark://anghabench-v1/8cc/extr_buffer.c_buf_append"
            # /usr/lib/gcc/x86_64-redhat-linux/8/../../../../lib64/crt1.o: In function `_start':
            # (.text+0x24): undefined reference to `main'
            # /tmp/benchmark-downloaded-36dcc2.o: In function `buf_append':
            # extr_buffer.c_buf_append.c:(.text+0x3e): undefined reference to `buf_write'
            # ====================================

            # chstone seems to work (.c is present)
            "benchmark://chstone-v0/adpcm",
            "benchmark://chstone-v0/aes",
            "benchmark://chstone-v0/blowfish",
            "benchmark://chstone-v0/dfadd",
            "benchmark://chstone-v0/dfdiv",
            "benchmark://chstone-v0/dfmul",
            "benchmark://chstone-v0/dfsin",
            "benchmark://chstone-v0/gsm",
            "benchmark://chstone-v0/jpeg",
            "benchmark://chstone-v0/mips",
            "benchmark://chstone-v0/motion",
            "benchmark://chstone-v0/sha",
        ]

        inc = 0
        for bench in benchmark_to_process:
            env.reset(benchmark=bench)

            for i in range(2):
                print("Main: step = ", i)
                observation, reward, done, info = env.step(
                    action=env.action_space.sample(),
                    observations=["runtime"],
                    rewards=["runtime"],
                )
                print(reward)
                # print(observation)
                print(info)
                # gf = pickle.loads(observation[0])
                # print(gf.tree(metric_column=reward_metric))
                # print(gf.dataframe[["line", "llvm_inst"]])

                # pdb.set_trace()
                if done:
                    env.reset()
            inc += 1
        print("I run %d benchmarks." % inc)


def just_a_tmp():
    with gym.make("llvm-v0") as env:
        print(type(env))
        env.reset()
        observation, reward, done, info = env.step(0)
        g = env.observation["Programl"]
        num_nodes = g.number_of_nodes()
        print(g)
        print(num_nodes)
        node = g.nodes[100]
        node["features"]["runtime"] = 33
        print(node["features"])


if __name__ == "__main__":
    # just_a_tmp()
    main()
