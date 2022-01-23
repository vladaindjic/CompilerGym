// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <fmt/format.h>
#include <glog/logging.h>

#include <boost/filesystem.hpp>
#include <iostream>

#include "compiler_gym/envs/llvm/service/BenchmarkFactory.h"
#include "compiler_gym/service/proto/compiler_gym_service.pb.h"
#include "compiler_gym/util/GrpcStatusMacros.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/SourceMgr.h"

namespace fs = boost::filesystem;

using namespace compiler_gym;
using namespace compiler_gym::llvm_service;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  CHECK(argc == 2) << "Usage: compute_observation <bitcode-path>";

  const fs::path workingDirectory{"."};

  compiler_gym::Benchmark request;
  request.set_uri("user");
  request.mutable_program()->set_uri(fmt::format("file:///{}", argv[1]));

  auto& benchmarkFactory = BenchmarkFactory::getSingleton(workingDirectory);
  std::unique_ptr<::llvm_service::Benchmark> benchmark;
  {
    const auto status = benchmarkFactory.getBenchmark(request, &benchmark);
    CHECK(status.ok()) << "Failed to load benchmark: " << status.error_message();
  }

  llvm::Module& module = benchmark->module();

  // Iterate through the functions in the module, removing the optnone attribute
  // where set.
  int removedOptNoneCount = 0;
  for (llvm::Function& function : module.functions()) {
    for (auto& attrSet : function.getAttributes()) {
      for (auto& attr : attrSet) {
        // NOTE(cummins): there is definitely a more efficient way of doing
        // this than string-ifying all of the attributes, but I don't know
        // it :-)
        if (attr.getAsString() == "optnone") {
          ++removedOptNoneCount;
          function.removeFnAttr(attr.getKindAsEnum());
        }
      }
    }
  }

  ASSERT_OK(benchmark->writeBitcodeToFile(argv[1]));
  std::cerr << "Stripped " << removedOptNoneCount << " optnone attributes from " << argv[1]
            << std::endl;

  return 0;
}
