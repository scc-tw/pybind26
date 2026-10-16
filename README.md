# pybind26

C++26 bindings for CPython 3.14. Functions and call policies are specialized at
compile time; a compiled runtime handles Python types and module state.

- Functions, classes, inheritance, callbacks, keyword arguments, and STL conversion.
- CPU arrays, NumPy, DLPack, `std::span`, and `std::mdspan` adapters.
- Optional C++26 reflection frontend.

## Requirements

- CMake 4.4+ and Ninja for presets.
- GCC 16 or Clang 22 with C++26 support, including `template for` and `std::expected`.
- CPython 3.14 development headers, including internal headers.

Extensions use the full CPython API and build with `-fno-exceptions`. Build GIL
(`cp314`) and free-threaded (`cp314t`) extensions separately. Native code must
synchronize shared mutable state.

## Usage

Place this repository at `external/pybind26` in your project.

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 4.4)
project(example LANGUAGES CXX)

add_subdirectory(external/pybind26)
pyb26_add_module(example SOURCES example.cpp)
```

`example.cpp`:

```cpp
#include <pyb26/pyb26.hpp>

int add(int a, int b) noexcept { return a + b; }

PYB26_MODULE(example, m) {
    m.def<&add>("add", pyb26::arg("a"), pyb26::arg("b"));
}
```

Build and import:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE=/path/to/python3.14
cmake --build build
PYTHONPATH=build /path/to/python3.14 -c 'import example; print(example.add(2, 3))'
# 5
```

Add `NUMPY` to `pyb26_add_module(...)` to compile the NumPy output adapter.
Use `PRIVATE_RUNTIME` when a module needs separate runtime compile settings.
Rebuild extensions when updating pybind26.

## Install

From this repository:

```sh
cmake --workflow --preset install-release
# Prefix: build/install/release
```

Consumers can replace `add_subdirectory(...)` with
`find_package(pyb26 CONFIG REQUIRED)`. Pass the installed prefix through
`-DCMAKE_PREFIX_PATH=/path/to/prefix` when configuring.

Use `install-debug` for a Debug package. Local compiler settings belong in
`CMakeUserPresets.json`.

## License

[MIT](LICENSE). Vendored DLPack retains its [Apache 2.0 license](include/pyb26/detail/vendor/dlpack/LICENSE).
