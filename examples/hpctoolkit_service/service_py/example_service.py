#! /usr/bin/env python3
#
#  Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""An example CompilerGym service in python."""
from bdb import set_trace
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

import programl as pg



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


def extractInstStr(ll_path: str) -> list:
    inst_list = []
    inst_list.append("dummy")

    with open(ll_path) as f:
        for line in f.readlines():
            if line[0:2] == '  ' and line[2] != ' ':
                print(len(inst_list), str(line))
                inst_list.append(str(line))    
    return inst_list


def addInstStrToDataframe(gf: ht.GraphFrame, ll_path: str) -> None:

    inst_list = extractInstStr(ll_path)  

    gf.dataframe["llvm_inst"] = ""

    for i, inst_idx in enumerate(gf.dataframe["line"]):
        if inst_idx < len(inst_list):
            gf.dataframe["llvm_inst"][i] = inst_list[inst_idx]


class HPCToolkitCompilationSession(CompilationSession):
    """Represents an instance of an interactive compilation session."""

    compiler_version: str = "1.0.0"

    action_spaces = [
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
                            "-O3",
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
            name="programl",
            binary_size_range=ScalarRange(
                min=ScalarLimit(value=0), max=ScalarLimit(value=1e5)
            ),
        ), 
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
                min=ScalarLimit(value=0), max=ScalarLimit(value=1e5)
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
        self._bc_path = str(self.working_dir / "benchmark.bc")
        self._exe_path = str(self.working_dir / "benchmark.exe")
        self._exe_struct_path = self._exe_path + ".hpcstruct"

        # Dump the benchmark source to disk.
        self._src_path = str(self.working_dir / "benchmark.c")
        with open(self.working_dir / "benchmark.c", "wb") as f:
            f.write(benchmark.program.contents)


        # Set run commands
        self.run_c = [self._exe_path]

        # Set compile commands
        if action_space.name == "llvm":
            self.compile_c = [
                [self._clang,"-o", self._llvm_path, "-S", "-emit-llvm", self._src_path],
                [self._opt, "--debugify", "-o", self._bc_path, self._llvm_path],
                [self._clang, self._bc_path, "-o", self._exe_path],
            ]

            self.compile_c_base = self.compile_c[:]
            self.compile_c_base[0].insert(1, "-O0")

        elif action_space.name == "gcc":
            self.compile_c = [
                ["gcc", "-g", "-o", self._exe_path, self._src_path]
            ]
            self.compile_c_base = self.compile_c[:]
            self.compile_c_base.insert(1, "-O0")
        else:
            print("Action space is doesn't exits: ", action_space)
            exit


        # Compile baseline at the beginning
        for cmd in self.compile_c_base:            
            run_command(
                cmd,
                timeout=30,
            )
            

        print(self.compile_c)

        print("\n", self.working_dir, "\n")
        pdb.set_trace()


    def apply_action(self, action: Action) -> Tuple[bool, Optional[ActionSpace], bool]:

        num_choices = len(self.action_spaces[0].choice[0].named_discrete_space.value)

        if len(action.choice) != 1:
            raise ValueError("Invalid choice count")

        choice_index = action.choice[0].named_discrete_value_index
        if choice_index < 0 or choice_index >= num_choices:
            raise ValueError("Out-of-range")


        # Compile benchmark with given optimization
        opt = self._action_space.choice[0].named_discrete_space.value[choice_index]
        logging.info(
            "Applying action %d, equivalent command-line arguments: '%s'",
            choice_index,
            opt,
        )

        self.compile_c[0].insert(1, opt)
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
            return Observation(scalar_double=0)
            # pdb.set_trace()

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
            print("\n", self.working_dir, "\n")


            hpctoolkit_cmd = [
                ["rm", "-rf", self._exe_struct_path, self.working_dir/"m", self.working_dir/"db"],
                ["hpcrun", "-e", "REALTIME@100","-t", "-o", self.working_dir/"m", self._exe_path],
                ["hpcstruct", "-o", self._exe_struct_path, self._exe_path],
                ["hpcprof-mpi", "-o", self.working_dir/"db", "--metric-db", "yes", "-S", self._exe_struct_path, self.working_dir/"m"]
            ]
            
            for cmd in hpctoolkit_cmd:
                print(cmd)

                run_command(
                    cmd,
                    timeout=30,
                )


            gf = parseHPCToolkit(str(self.working_dir/"db"))          
            addInstStrToDataframe(gf, self._llvm_path)

            pickled = pickle.dumps(gf)
            return Observation(binary_value=pickled)

        elif "programl":

            with open(self._src_path, "r") as f:
                code_str = f.read().rstrip()

                G = pg.from_cpp(code_str)
                G = pg.to_networkx(G)
                
                # for node in G.adjacency():
                for n_id in G.nodes():
                    node = G.nodes[n_id]
                    ll_str = node['features']['full_text'][0] if 'features' in node else "No features"
                    print(node)
                    # print('ll_str = ', ll_str)
                    pdb.set_trace()

            return 0

        else:
            raise KeyError(observation_space.name)


if __name__ == "__main__":
    create_and_run_compiler_gym_service(HPCToolkitCompilationSession)
