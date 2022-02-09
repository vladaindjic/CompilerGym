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

import compiler_gym.third_party.llvm as llvm
from compiler_gym import site_data_path
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
from compiler_gym.util.commands import run_command, Popen


def run_command_stdout_redirect(cmd: List[str], timeout: int, output_file):
    with Popen(
        cmd, stdout=output_file, stderr=subprocess.PIPE, universal_newlines=True
    ) as process:
        stdout, stderr = process.communicate(timeout=timeout)
        if process.returncode:
            returncode = process.returncode
            try:
                # Try and decode the name of a signal. Signal returncodes
                # are negative.
                returncode = f"{returncode} ({Signals(abs(returncode)).name})"
            except ValueError:
                pass
            raise OSError(
                f"Compilation job failed with returncode {returncode}\n"
                f"Command: {' '.join(cmd)}\n"
                f"Stderr: {stderr.strip()}"
            )


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
                min=ScalarLimit(value=0), max=ScalarLimit(value=1e5)
            ),
        ),
    ]

    def proto_buff_container_to_list(self, container):
        # Copy proto buff container to python list.
        compile_cmd = [el for el in container]
        # NOTE: if run_command function can work with proto buf containers,
        # then generating the list can be omitted. Uncomment the following statement instead.
        # compile_cmd = build_cmd.argument
        return compile_cmd

    def prepare_build_cmd(self, build_cmd):
        """
        build_cmd is in the following format.
        build_cmd {
          argument: "$CC"
          argument: "$IN"
          argument: "-lm"
          timeout_seconds: 60
          outfile: "a.out"
        }
        Only argument list with the following form: ["$CC", "$IN", "-lm"] is processed.
        """
        compile_cmd = self.proto_buff_container_to_list(build_cmd.argument)
        # Just for the debugging purposes.
        assert compile_cmd[0] == "$CC"
        assert compile_cmd[1] == "$IN"
        # Replace $CC with the llvm_clang.
        compile_cmd[0] = self._clang
        # Replace $IN with path to the .bc files
        compile_cmd[1] = self._bc_path
        # Append the output file of the compile command.
        compile_cmd.extend(["-o", self._exe_path])
        # Add the last compile command.
        self.compile_c.append(compile_cmd)

    def prepare_pre_run_cmd(self, pre_run_cmd):
        """
        pre_run_cmd is in the following format
        pre_run_cmd {
          argument: "echo"
          argument: "1"
          argument: ">_finfo_dataset"
          timeout_seconds: 30
        }
        Only argument is considered for now.
        """
        pre_run_cmd = self.proto_buff_container_to_list(pre_run_cmd.argument)
        # append pre_run_cmd list to the pre_run
        self.pre_run.append(pre_run_cmd)

    def pre_run_cmd_with_redirection(self, cmd):
        """
        Pre run command might redirect the stdout to a file.
        Try to recignize this.
        """
        # FIXME: This function assumes that the last argument may contain the files
        #   to which the stdout should be redirected.
        if cmd[-1].startswith('>'):
            # Remove '>' from file name and prepend working_dir path
            file_path = self.working_dir / cmd[-1][1:]
            with open(file_path, 'w') as f:
                # Remove the stdout redirection from the cmd.
                cmd = cmd[:-1]
                run_command_stdout_redirect(cmd, timeout=30, output_file=f)
        else:
            # Execute the command with no redirection
            run_command(cmd, timeout=30)

    def prepare_run_cmd(self, run_cmd):
        """
        run_cmd is in the following format:
        run_cmd {
          argument: "./a.out"
          argument: "/home/vi3/.local/share/compiler_gym/llvm-v0/cbench-v1-runtime-data/runtime_data/automotive_qsort_data/1.dat"
          timeout_seconds: 300
          infile: "a.out"
          infile: "_finfo_dataset"
          outfile: "sorted_output.dat"
        }
        Only argument is considered for now.
        """
        print("________ Run command: \n", run_cmd)
        run_cmd = self.proto_buff_container_to_list(run_cmd.argument)

        # Just for the debugging purposes.
        assert run_cmd[0] == './a.out'
        # Replace ./a.out with the _exe_path
        run_cmd[0] = self._exe_path
        self.run_c = run_cmd

    def prepare_cbench(self, benchmark: Benchmark):
        self._running_cbench = True
        # Extract build command.
        self.prepare_build_cmd(benchmark.dynamic_config.build_cmd)
        # Extract all pre run commands
        for cmd in benchmark.dynamic_config.pre_run_cmd:
            self.prepare_pre_run_cmd(cmd)
        # Extract run command.
        self.prepare_run_cmd(benchmark.dynamic_config.run_cmd)

    def prepare_csmith(self, benchmark: Benchmark):
        # Extract build command.
        self.prepare_build_cmd(benchmark.dynamic_config.build_cmd)
        # Pre run command should not exist.
        # Just for the debugging purposes.
        assert len(benchmark.dynamic_config.pre_run_cmd) == 0
        # Extract run command.
        self.prepare_run_cmd(benchmark.dynamic_config.run_cmd)

    def prepare_other_bench(self, benchmark: Benchmark):
        # Compile the .bc to executable.
        self.compile_c.append([
            self._clang,
            self._bc_path,
            "-o", self._exe_path,
            # "-lm",
            # "-nostartfiles"
        ])
        # Run with no additional arguments.
        self.run_c = [self._exe_path]

    def prepare_bitcode_bench(self, benchmark: Benchmark):
        print(benchmark.uri)
        # Dump downloaded .bc file.
        with open(self._bc_download_path, 'wb') as f:
            f.write(benchmark.program.contents)
        # Command that converts .bc to .ll
        self.compile_c.append(
            [self._llvm_dis, "-o", self._llvm_path, self._bc_download_path],
        )
        # Command that applies opt with --debugify argument.
        self.compile_c.append(
            [self._opt, "--debugify", "-o", self._bc_path, self._llvm_path],
        )
        if "benchmark://cbench-v1" in str(benchmark.uri):
            # Extract compile and exec command dedicated to the benchmakr itself.
            self.prepare_cbench(benchmark)
        elif "generator://csmith-v0" in str(benchmark.uri):
            self.prepare_csmith(benchmark)
        else:
            self.prepare_other_bench(benchmark)

    def prepare_bench(self, benchmark: Benchmark):
        # pdb.set_trace()
        print(benchmark.uri)
        if benchmark.program.contents.startswith(b'BC'):
            self.prepare_bitcode_bench(benchmark)
        elif "mibench" in str(benchmark.uri):
            self.prepare_bitcode_bench(benchmark)
        elif "opencv" in str(benchmark.uri):
            self.prepare_bitcode_bench(benchmark)
        elif "tensorflow" in str(benchmark.uri):
            self.prepare_bitcode_bench(benchmark)
        else:
            return -1

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
        self._llvm_dis = str(llvm.llvm_dis_path())
        self._llc = str(llvm.llc_path())
        self._llvm_diff = str(llvm.llvm_diff_path())
        self._opt = str(llvm.opt_path())

        self._llvm_path = str(self.working_dir / "benchmark.ll")
        self._llvm_before_path = str(self.working_dir / "benchmark.previous.ll")
        self._bc_path = str(self.working_dir / "benchmark.bc")
        # vi3: Used only if the benchmark is downloaded in .bc format.
        self._bc_download_path = str(self.working_dir / "benchmark-downloaded.bc")
        self._exe_path = str(self.working_dir / "benchmark.exe")
        self._exe_struct_path = self._exe_path + ".hpcstruct"

        self.compile_c = []
        self.pre_run = []
        self.run_c = None

        self._running_cbench = False

        if self.prepare_bench(benchmark) != -1:
            # FIXME: just a stupid workaround.
            pass
        else:
            # Dump the benchmark source to disk.
            self._src_path = str(self.working_dir / "benchmark.c")
            with open(self.working_dir / "benchmark.c", "wb") as f:
                f.write(benchmark.program.contents)
            # generate .ll from .c (source code)
            # vi3: Is problem if -O0 is supplied here?
            self.compile_c = [
                [self._clang, "-O0", "-o", self._llvm_path, "-S", "-emit-llvm", self._src_path],
                [self._opt, "--debugify", "-o", self._bc_path, self._llvm_path],
                [self._clang, self._bc_path, "-o", self._exe_path]
            ]
            # Set run commands
            self.run_c = [self._exe_path]

        # Set compile commands
        if action_space.name == "llvm":
            self.compile_c_base = self.compile_c[:]
            # vi3: I'm not sure if we need this for .ll generated from .bc by using llvm-dis.
            # self.compile_c_base[0].insert(1, "-O0")

        elif action_space.name == "gcc":
            self.compile_c = [
                ["gcc", "-g", "-o", self._exe_path, self._src_path]
            ]
            self.compile_c_base = self.compile_c[:]
            self.compile_c_base.insert(1, "-O0")
        else:
            print("Action space is doesn't exits: ", action_space)
            exit

        # pdb.set_trace()
        # Compile baseline at the beginning
        for cmd in self.compile_c_base:
            run_command(
                cmd,
                timeout=30,
            )
            # pdb.set_trace()

        for cmd in self.pre_run:
            # The stdout may be redirected to a file.
            self.pre_run_cmd_with_redirection(cmd)

        print(self.compile_c)

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

        # FIXME vi3: see to which compile command "opt" should be applied.
        #  The following line assumes that the first command compiles .c to .ll,
        #  which is not the case if the benchmark is in .bc format.
        # self.compile_c[0].insert(1, opt)
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

            # TODO: add documentation that benchmarks need print out execution time
            # Running 5 times and taking the average of middle 3
            exec_times = []
            print("I need to run this: ", self.run_c)
            # For some reason, we need to change pwd when runnin cbench data sets
            if self._running_cbench:
                os.chdir(self.working_dir)
            # pdb.set_trace()
            for _ in range(5):
                stdout = run_command(
                    self.run_c,
                    timeout=30,
                )

                print(">>> The stdout of the run with the size:", len(stdout))
                max_out = len(stdout) if len(stdout) < 100 else 100
                if max_out > 0:
                    print(">>> Showing some stdout\n", stdout[:100])
                # pdb.set_trace()
                try:
                    # exec_times.append(int(stdout))
                    exec_times.append(10)
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
            # print("get_observation: hpctoolkit")
            # print("\n", self.working_dir, "\n")
            #
            #
            # hpctoolkit_cmd = [
            #     ["rm", "-rf", self._exe_struct_path, self.working_dir/"m", self.working_dir/"db"],
            #     ["hpcrun", "-e", "REALTIME@100", "-o", self.working_dir/"m", self._exe_path],
            #     ["hpcstruct", "-o", self._exe_struct_path, self._exe_path],
            #     ["hpcprof", "-o", self.working_dir/"db", "--metric-db", "yes", "-S", self._exe_struct_path, self.working_dir/"m"]
            # ]
            #
            # for cmd in hpctoolkit_cmd:
            #     print(cmd)
            #     # pdb.set_trace()
            #
            #     run_command(
            #         cmd,
            #         timeout=30,
            #     )
            #
            # # pdb.set_trace()
            #
            # gf = parseHPCToolkit(str(self.working_dir/"db"))
            # addInstStrToDataframe(gf, self._llvm_path)
            #
            # pickled = pickle.dumps(gf)
            return Observation(1)

        else:
            raise KeyError(observation_space.name)


if __name__ == "__main__":
    create_and_run_compiler_gym_service(HPCToolkitCompilationSession)
