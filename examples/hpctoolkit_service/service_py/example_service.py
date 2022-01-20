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
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import hatchet as ht
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


def parseHPCToolkit(db_path: str) -> ht.GraphFrame:

    # Use hatchet's ``from_hpctoolkit`` API to read in the HPCToolkit database.
    # The result is stored into Hatchet's GraphFrame.
    return ht.GraphFrame.from_hpctoolkit(db_path)


class HPCToolkitCompilationSession(CompilationSession):
    """Represents an instance of an interactive compilation session."""

    compiler_version: str = "1.0.0"

    action_spaces = [
        # ActionSpace(
        #     name="gcc",
        #     choice=[
        #         ChoiceSpace(
        #             name="optimization_choice",
        #             named_discrete_space=NamedDiscreteSpace(
        #                 value=[
        #                     "-O0",
        #                     "-O1",
        #                     "-O2",
        #                 ],
        #             ),
        #         )
        #     ],
        # ),
        ActionSpace(
            name="llvm",
            choice=[
                ChoiceSpace(
                    name="optimization_choice",
                    named_discrete_space=NamedDiscreteSpace(
                        value=[
                            "-O0",
                            "-O1",
                            "-O2",
                        ],
                    ),
                )
            ],
        ),
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
            name="hpctoolkit",
            binary_size_range=ScalarRange(
                min=ScalarLimit(value=5), max=ScalarLimit(value=5)
            ),
        ),
    ]

    def __init__(
        self,
        working_directory: Path,
        action_space: ActionSpace,
        benchmark: Benchmark,  # TODO: Dejan use Benchmark rather than hardcoding benchmark path here!
        # use_custom_opt: bool = True,
    ):
        super().__init__(working_directory, action_space, benchmark)
        logging.info("Started a compilation session for %s", benchmark.uri)
        self._benchmark = benchmark
        self._action_space = action_space

        # Resolve the paths to LLVM binaries once now.
        self._clang = str(llvm.clang_path())
        self._llc = str(llvm.llc_path())
        self._llvm_diff = str(llvm.llvm_diff_path())
        self._opt = str(llvm.opt_path())

        self._llvm_path = str(self.working_dir / "benchmark.ll")
        self._llvm_before_path = str(self.working_dir / "benchmark.previous.ll")
        self._obj_path = str(self.working_dir / "benchmark.o")
        self._exe_path = str(self.working_dir / "benchmark.exe")

        # Dump the benchmark source to disk.
        self._src_path = str(self.working_dir / "benchmark.c")
        with open(self.working_dir / "benchmark.c", "wb") as f:
            f.write(benchmark.program.contents)

        self.benchmark_c_path: Path = "/home/dx4/tools/CompilerGym/examples/hpctoolkit_service/benchmarks/cpu-benchmarks/conv2d.c"
        self.benchmark_c_exec: Path = "/home/dx4/tools/CompilerGym/examples/hpctoolkit_service/benchmarks/cpu-benchmarks/conv2d"
        self.benchmark_c_ll: Path = "/home/dx4/tools/CompilerGym/examples/hpctoolkit_service/benchmarks/cpu-benchmarks/conv2d.ll"
        self.benchmark_c_bc: Path = "/home/dx4/tools/CompilerGym/examples/hpctoolkit_service/benchmarks/cpu-benchmarks/conv2d.bc"
        self.benchmark_c_dir: Path = os.path.dirname(self.benchmark_c_path)

        self.clang_cmd = [
            [
                "clang",
                "-o",
                self.benchmark_c_ll,
                "-S",
                "-emit-llvm",
                self.benchmark_c_path,
            ],
            ["opt", "--debugify", "-o", self.benchmark_c_bc, self.benchmark_c_ll],
            ["clang", self.benchmark_c_bc, "-o", self.benchmark_c_exec],
        ]

        if action_space.name == "llvm":
            self.compile_c = self.clang_cmd
            self.compile_c_base = self.compile_c[:]
            self.compile_c_base[0].insert(1, "-O0")

        elif action_space.name == "gcc":
            self.compile_c = [
                [
                    "gcc",
                    "-g",
                    "-o",
                    self.benchmark_c_exec,
                    self.benchmark_c_path,
                ]
            ]
            self.compile_c_base = self.compile_c[:]
            self.compile_c_base.insert(1, "-O0")
        else:
            print("Action space is doesn't exits: ", action_space)
            exit

        # Clean benchmark directory

        # clean_ben_dir_com = [
        #     "rm",
        #     "-rf",
        #     self.benchmark_c_dir +"/db",
        #     self.benchmark_c_dir +"/m",
        #     self.benchmark_c_dir +"/*.hpcstruct",
        #     self.benchmark_c_exec,
        #     ]

        # pdb.set_trace()
        # print(clean_ben_dir_com)
        # run_command(
        #     clean_ben_dir_com,
        #     timeout=30,
        # )

        for cmd in self.compile_c_base:
            stdout = run_command(
                cmd,
                timeout=30,
            )
            print(stdout)

        self.run_c = [self.benchmark_c_exec]
        print(self.compile_c)
        pdb.set_trace()

    def apply_action(self, action: Action) -> Tuple[bool, Optional[ActionSpace], bool]:

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
        self.compile_c[0].insert(1, opt)

        # Compile with action (gcc, llvm)
        # os.chdir(os.path.dirname(self.benchmark_c_path))

        for cmd in self.compile_c:
            run_command(
                cmd,
                timeout=30,
            )

        # TODO: Dejan properly implement these
        action_had_no_effect = False
        end_of_session = False
        new_action_space = None
        return (end_of_session, new_action_space, action_had_no_effect)

    def get_observation(self, observation_space: ObservationSpace) -> Observation:
        # #pdb.set_trace()
        logging.info("Computing observation from space %s", observation_space.name)

        if observation_space.name == "runtime":
            print("get_observation: runtime")
            pdb.set_trace()

            # os.chdir(os.path.dirname(self.benchmark_c_path))

            # TODO: add documentation that benchmarks need print out execution time
            # Running 5 times and taking the average of middle 3
            exec_times = []
            for _ in range(5):
                stdout = run_command(
                    self.run_c,
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

        elif observation_space.name == "hpctoolkit":
            print("get_observation: hpctoolkit")
            pdb.set_trace()
            os.chdir(os.path.dirname(self.benchmark_c_path))

            # os.system(
            #     "gcc -O3 -g -o " + self.benchmark_c_exec + " " + self.benchmark_c_path
            # )

            for cmd in self.clang_cmd:
                print(cmd)
                pdb.set_trace()
                run_command(
                    cmd,
                    timeout=30,
                )

            os.system("rm -rf m* db*")
            os.system("hpcrun -e REALTIME@100 -o m " + self.benchmark_c_exec)
            os.system("hpcstruct " + self.benchmark_c_exec)
            os.system(
                "hpcprof-mpi -o db --metric-db yes -S "
                + self.benchmark_c_exec
                + ".hpcstruct m"
            )

            gf = parseHPCToolkit("db")
            print(gf.dataframe)
            pdb.set_trace()

            pickled = pickle.dumps(gf)
            return Observation(binary_value=pickled)

        else:
            raise KeyError(observation_space.name)


if __name__ == "__main__":
    create_and_run_compiler_gym_service(HPCToolkitCompilationSession)
