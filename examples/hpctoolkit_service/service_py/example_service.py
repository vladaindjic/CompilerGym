#! /usr/bin/env python3
#
#  Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""An example CompilerGym service in python."""
import logging
import os
import pdb
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import utils

import compiler_gym.third_party.llvm as llvm
from compiler_gym.service import CompilationSession
from compiler_gym.service.proto import (
    Action,
    ActionSpace,
    Benchmark,
    ChoiceSpace,
    NamedDiscreteSpace,
    Observation,
    ObservationSpace,
    ScalarLimit,
    ScalarRange,
    ScalarRangeList,
)
from compiler_gym.service.runtime import create_and_run_compiler_gym_service
from compiler_gym.util.commands import run_command


class HPCToolkitCompilationSession(CompilationSession):
    """Represents an instance of an interactive compilation session."""

    compiler_version: str = "1.0.0"

    # The action spaces supported by this service. Here we will implement a
    # single action space, called "default", that represents a command line with
    # three options: "a", "b", and "c".
    action_spaces = [
        ActionSpace(
            name="hpctoolkit",
            choice=[
                ChoiceSpace(
                    name="optimization_choice",
                    named_discrete_space=NamedDiscreteSpace(
                        value=[
                            "-Xptxas -O0",
                            "-Xptxas -O1",
                            "-Xptxas -O2",
                        ],
                    ),
                )
            ],
        )
    ]

    # A list of observation spaces supported by this service. Each of these
    # ObservationSpace protos describes an observation space.
    observation_spaces = [
        ObservationSpace(
            name="runtime",
            scalar_double_range=ScalarRange(min=ScalarLimit(value=0)),
            deterministic=False,
            platform_dependent=True,
            default_value=Observation(
                scalar_double=0,
            ),
        ),
        ObservationSpace(
            name="hatchet",
            scalar_double_range=ScalarRange(min=ScalarLimit(value=0)),
            deterministic=False,
            platform_dependent=True,
            default_value=Observation(
                scalar_double=0,
            ),
        ),
    ]

    def __init__(
        self,
        working_directory: Path,
        action_space: ActionSpace,
        benchmark: Benchmark,
        # use_custom_opt: bool = True,
    ):
        super().__init__(working_directory, action_space, benchmark)
        logging.info("Started a compilation session for %s", benchmark.uri)
        self._benchmark = benchmark
        self._action_space = action_space

        # Dump the benchmark source to disk.
        with open(self.working_dir / "benchmark.c", "wb") as f:
            f.write(benchmark.program.contents)

        self._llvm_path = str(self.working_dir / "benchmark.ll")
        self._llvm_before_path = str(self.working_dir / "benchmark.previous.ll")
        self._obj_path = str(self.working_dir / "benchmark.o")
        self._exe_path = str(self.working_dir / "benchmark.exe")

        self.benchmark_cuda_path: Path = "/home/dx4/tools/CompilerGym/examples/hpctoolkit_service/benchmarks/cuda-runtime/main.cu"
        self.benchmark_cuda_exec: Path = "/home/dx4/tools/CompilerGym/examples/hpctoolkit_service/benchmarks/cuda-runtime/main"

        self.compile_nvcc = [
            "/usr/local/cuda-11.4/bin/nvcc",
            "-o",
            "main",
            "-Xcompiler",
            "-g",
            "-O3",
            "-lineinfo",
            "-arch",
            "sm_70",
            "-Xptxas",
            "-O0",
            "-lcudart",
            "-lcuda",
            "-lstdc++",
            "-lm",
            self.benchmark_cuda_path,
        ]
        self.run_nvcc = [self.benchmark_cuda_exec]

        self.benchmark_c_path: Path = "/home/dx4/tools/CompilerGym/examples/hpctoolkit_service/benchmarks/cpu-benchmarks/conv2d.c"
        self.benchmark_c_exec: Path = "/home/dx4/tools/CompilerGym/examples/hpctoolkit_service/benchmarks/cpu-benchmarks/conv2d"

        # self.compile_c = "gcc -O3 -g -o " + self.benchmark_c_exec + " " + self.benchmark_c_path

        self.compile_c = [
            "gcc",
            "-O3",
            "-g",
            "-o",
            self.benchmark_c_exec,
            self.benchmark_c_path,
        ]

        self.run_c = [
            "hpctoolkit",
            "-o",
            "m",
            self.benchmark_c_exec,
        ]

    def apply_action(self, action: Action) -> Tuple[bool, Optional[ActionSpace], bool]:
        # #pdb.set_trace()

        num_choices = len(self.action_spaces[0].choice[0].named_discrete_space.value)

        if len(action.choice) != 1:
            raise ValueError("Invalid choice count")

        choice_index = action.choice[0].named_discrete_value_index
        logging.info("Applying action %d", choice_index)

        if choice_index < 0 or choice_index >= num_choices:
            raise ValueError("Out-of-range")

        # Here is where we would run the actual action to update the environment's
        # state.

        opt = self._action_space.choice[0].named_discrete_space.value[choice_index]
        logging.info(
            "Applying action %d, equivalent command-line arguments: '%s'",
            choice_index,
            opt,
        )

        # #pdb.set_trace()

        # run_command(
        #     self.compile_nvcc,
        #     timeout=30,
        # )

        # TODO: Dejan properly implement these
        action_had_no_effect = False
        end_of_session = False
        new_action_space = None
        return (end_of_session, new_action_space, action_had_no_effect)

    def get_observation(self, observation_space: ObservationSpace) -> Observation:
        # #pdb.set_trace()

        logging.info("Computing observation from space %s", observation_space.name)

        if observation_space.name == "runtime":
            pdb.set_trace()

            os.chdir(os.path.dirname(self.benchmark_cuda_path))

            run_command(
                self.compile_nvcc,
                timeout=30,
            )
            # TODO: add documentation that benchmarks need print out execution time
            # Running 5 times and taking the average of middle 3
            exec_times = []
            for _ in range(5):
                stdout = run_command(
                    self.run_nvcc,
                    timeout=30,
                )
                print(stdout)
                try:
                    exec_times.append(int(stdout))
                except ValueError:
                    raise ValueError(
                        f"Error in parsing execution time from output of command\n"
                        f"Please ensure that the source code of the benchmark measures execution time and prints to stdout\n"
                        f"Stdout of the program: {stdout}"
                    )

            exec_times = np.sort(exec_times)
            avg_exec_time = np.mean(exec_times[1:4])

            return Observation(scalar_double=avg_exec_time)

        elif observation_space.name == "hatchet":
            pdb.set_trace()
            os.chdir(os.path.dirname(self.benchmark_c_path))

            os.system(
                "gcc -O3 -g -o " + self.benchmark_c_exec + " " + self.benchmark_c_path
            )

            os.system("rm -rf m* db*")
            os.system("hpcrun -o m " + self.benchmark_c_exec)
            os.system("hpcstruct " + self.benchmark_c_exec)
            os.system(
                "hpcprof-mpi -o db --metric-db yes -S "
                + self.benchmark_c_exec
                + ".hpcstruct m"
            )
            os.system(
                "python /home/dx4/tools/CompilerGym/examples/hpctoolkit_service/parser/parsedb.py db"
            )

            return Observation(scalar_double=0)

        else:
            raise KeyError(observation_space.name)


if __name__ == "__main__":
    create_and_run_compiler_gym_service(HPCToolkitCompilationSession)
