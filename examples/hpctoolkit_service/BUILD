# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "hpctoolkit_service",
    srcs = ["__init__.py"],
    data = [
        "//examples/hpctoolkit_service/benchmarks/cpu-benchmarks",
        "//examples/hpctoolkit_service/service_py:hpctoolkit-service-py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/envs/llvm",
        "//compiler_gym/util",
    ],
)

py_test(
    name = "env_tests",
    srcs = ["env_tests.py"],
    deps = [
        ":hpctoolkit_service",
        "//compiler_gym",
        "//tests:test_main",
    ],
)

py_binary(
    name = "example",
    srcs = ["example.py"],
    deps = [
        ":hpctoolkit_service",
        "//compiler_gym",
    ],
)
