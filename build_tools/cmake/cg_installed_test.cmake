# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# cg_add_installed_test()
#
# Creates a build-time and exported install-time test. All tests are installed
# into the tests/ tree. Calling code must arrange to install dependencies of the
# test into that tree.
#
# Parameters:
# TEST_NAME: Name of the test (as in "some/path/to/test").
# COMMAND: Passed to add_test() as is.
# ENVIRONMENT: Set as the ENVIRONMENT property of the build-time test.
# INSTALLED_COMMAND: Corrollary to the 'COMMAND' argument but added to the
#   install time definition.
# WORKING_DIRECTORY: Passed to add_test() as is. Note that in the install tree
# all tests run in the tests/ directory.
# LABELS: Labels to pass to add_test() and installed tests.
function(cg_add_installed_test)
  cmake_parse_arguments(
    _RULE
    ""
    "TEST_NAME"
    "COMMAND;ENVIRONMENT;INSTALLED_COMMAND;WORKING_DIRECTORY;LABELS"
    ${ARGN}
  )


  add_test(
    NAME
      ${_RULE_TEST_NAME}
    COMMAND
      ${_RULE_COMMAND}
  )
  if (DEFINED _RULE_WORKING_DIRECTORY)
    set_property(
      TEST
        ${_RULE_TEST_NAME}
      PROPERTY WORKING_DIRECTORY
        "${_RULE_WORKING_DIRECTORY}"
    )
  endif()
  set_property(
    TEST
      ${_RULE_TEST_NAME}
    PROPERTY LABELS
      "${_RULE_LABELS}"
  )
  set_property(
    TEST
      ${_RULE_TEST_NAME}
    PROPERTY ENVIRONMENT
      "TEST_TMPDIR=${CMAKE_BINARY_DIR}/${_RULE_TEST_NAME}_test_tmpdir"
      ${_RULE_ENVIRONMENT}
  )
  cg_add_test_environment_properties(${_RULE_TEST_NAME})

  # Write the to the installed ctest file template.
  set(_installed_ctest_input_file
        "${CMAKE_BINARY_DIR}/cg_installed_tests.cmake.in")
  get_property(_has_tests GLOBAL PROPERTY COMPILER_GYM_HAS_INSTALLED_TESTS)
  if(NOT _has_tests)
    # First time.
    file(WRITE "${_installed_ctest_input_file}")  # Truncate.
    set_property(GLOBAL PROPERTY COMPILER_GYM_HAS_INSTALLED_TESTS ON)
  endif()

  # Now write directives to the installed tests cmake file.
  file(APPEND "${_installed_ctest_input_file}"
    "add_test(${_RULE_TEST_NAME} ${_RULE_INSTALLED_COMMAND})\n"
    "set_tests_properties(${_RULE_TEST_NAME} PROPERTIES LABELS \"${_RULE_LABELS}\")\n"
  )

  # First time generation and setup to install. Note that since this all runs
  # at the generate phase, it doesn't matter that we trigger it before all
  # tests accumulate.
  if(NOT _has_tests)
    set(_installed_ctest_output_file "${CMAKE_BINARY_DIR}/cg_installed_tests.cmake")
    file(GENERATE
      OUTPUT "${_installed_ctest_output_file}"
      INPUT "${_installed_ctest_input_file}"
    )
    install(FILES "${_installed_ctest_output_file}"
      DESTINATION tests
      RENAME "CTestTestfile.cmake"
      COMPONENT Tests
    )
  endif()
endfunction()
