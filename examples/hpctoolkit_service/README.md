# HPCToolkit service

CompilerGym uses a
[client/service architecture](https://facebookresearch.github.io/CompilerGym/compiler_gym/service.html).
The client is the frontend python API and the service is the backend that
handles the actual compilation. The backend exposes a
[CompilerGymService](https://github.com/facebookresearch/CompilerGym/blob/development/compiler_gym/service/proto/compiler_gym_service.proto)
RPC interface that the frontend interacts with.

This directory contains an example backend service that runs GPU code and collect dynamic features. Initially it supports only execution time as feature, but it will support other dynamic features like GPU pc samples and hardware counters.

If you have any questions please [file an
issue](https://github.com/facebookresearch/CompilerGym/issues/new/choose).


## Features

* A action space consisits of LLVM optimizations: `["-O0", "-O1", "-O2"]`. The action space
  never changes. Actions never end the episode.
* There are one observation spaces:
  * `runtime` which returns the float value.


## Implementation

Only python service is supported [service_py/example_service.py](service_py/example_service.py). The module [__init__.py](__init__.py) defines the reward space, dataset, and registers two new environments using these services.

The file [demo.py](demo.py) demonstrates how to use these example environments
using CompilerGym's bazel build system. The file [env_tests.py](env_tests.py)
contains unit tests for the example services. Because the Python and C++
services implement the same interface, the same tests are run against both
environments.

## Usage

Run the demo script using:

```sh
$ bazel run -c opt //examples/hpctoolkit_service:demo
```

Run the unit tests using:

```sh
$ bazel test //examples/hpctoolkit_service/...
```

### Using the python service without bazel

Because the python service contains no compiled code, it can be run directly as
a standalone script without using the bazel build system. From the root of the
CompilerGym repository, run:

```sh
$ cd examples
$ python3 hpctoolkit_service/demo_without_bazel.py
```
