# Build

## Relinker only

The relinker can be built without initializing submodules or configuring SDL, Vulkan, FFmpeg, FreeType or the shader recompiler. It requires CMake, a C++20 compiler and a build tool. Python 3 enables the Python regression tests.

```sh
cmake -S . -B build-relinker -G Ninja -DCMAKE_BUILD_TYPE=Release -DANYPS5_RELINKER_ONLY=ON -DBUILD_TESTING=ON
cmake --build build-relinker --parallel
ctest --test-dir build-relinker --output-on-failure
```

This mode builds the conversion tool and its tests on Linux, Windows and macOS, including Apple Silicon. macOS uses AppleClang from the Xcode command-line tools; Windows uses the MinGW-w64 toolchain below. The executable is `build-relinker/core/relinker/relinker` (`relinker.exe` on Windows with Ninja).

The output remains x86-64 Linux ELF or Windows PE. Converted games need system libraries built for the target OS and a compatible x86-64 host. This mode does not build those libraries or provide macOS game execution. Tests inspect both output formats; execution checks run only on their compatible hosts.

Use a separate build directory for the full build.

## Full build

```sh
git submodule update --init --recursive
```

## Requirements

- x86-64, Git, CMake 3.22.1 or newer, Ninja, C++20.
- Linux: GCC, G++, binutils. SDL's X11 backend requires X11 and Xext development headers (`libx11-dev` and `libxext-dev` on Debian/Ubuntu).
- Windows: only MinGW-w64 GCC 15.2.0 (WinLibs `x86_64-ucrt-posix-seh`, release `15.2.0posix-14.0.0-ucrt-r7`) is currently supported. Add its `mingw64/bin` directory to `PATH` before configuring.
- FFmpeg binaries are downloaded during configuration unless `FFMPEG_PREBUILT_DIR` is set. With the WinLibs CMake, the download fails with status 60 (`SSL peer certificate or SSH remote key was not OK`) unless `SSL_CERT_FILE` names a CA bundle, for example `C:\Program Files\Git\mingw64\etc\ssl\certs\ca-bundle.crt` from Git for Windows, as in CI.

## Commands

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build build --parallel
cmake --build build --target libs --parallel
```

`libs` is a custom target: every library under [core/libs/prx](../../core/libs/prx) is built with
`EXCLUDE_FROM_ALL`, so the first command alone does not produce them. Titles load the patched `.prx`
files from `build/core/libs/libs`, which only that second step refreshes. Running a title after a
library change without it therefore tests the previous binaries and can show no effect at all.

[Relinker usage and runtime layout](../user/USAGE.md).

## CMake flags

Project switches accept `ON` or `OFF`:

| Flag                             | Default | Effect                                           |
|----------------------------------|---------|--------------------------------------------------|
| `-DBUILD_TESTING=ON`             | `OFF`   | Build and register tests.                        |
| `-DANYPS5_RELINKER_ONLY=ON`      | `OFF`   | Build only the relinker and its tests, without third-party dependencies. |
| `-DANYPS5_ENABLE_SPIRV_TOOLS=ON` | `OFF`   | Enable SPIR-V validation and optimization.       |
| `-DAPS5_ENABLE_TIMING_LOG=ON`    | `OFF`   | Compile frame timing logging.                    |
| `-DAPS5_AGC_CREATE_LOG=OFF`      | `ON`    | Disable successful `sceAgcCreateShader` logging. |
| `-DAGC_BUILD_VISUAL_TEST=ON`     | `OFF`   | Build the standalone AGC SPIR-V visual test.     |

Build configuration parameters:

| Flag                                   | Value                                                              |
|----------------------------------------|--------------------------------------------------------------------|
| `-DCMAKE_BUILD_TYPE=Release`           | `Debug`, `Release`, `RelWithDebInfo`, or `MinSizeRel`.             |
| `-DCMAKE_C_COMPILER=gcc`               | C compiler name or absolute path.                                  |
| `-DCMAKE_CXX_COMPILER=g++`             | C++ compiler name or absolute path.                                |
| `-DCMAKE_C_COMPILER_LAUNCHER=ccache`   | Optional C compiler cache; requires `ccache`.                      |
| `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache` | Optional C++ compiler cache; requires `ccache`.                    |
| `-DFFMPEG_PREBUILT_DIR=<path>`         | Unpacked FFmpeg package for the target platform; empty by default. |

SDL and FreeType settings forced by the root `CMakeLists.txt` cannot be overridden with `-D`.

## Pipeline statistics

Set `APS5_PIPELINE_STATS=1` to capture and print driver statistics for each newly created graphics or compute pipeline. This requires `VK_KHR_pipeline_executable_properties` and `pipelineExecutableInfo`; an unsupported device fails with an error. Statistic names and units are driver-specific. Capturing statistics can increase pipeline compilation cost. The setting is disabled by default.

## Motion emulation

Set `APS5_PAD_MOTION_EMULATION=1` to give a controller without an accelerometer or gyroscope emulated motion (a controller with both sensors is not affected). While the right stick click is held:

- the right stick tilts the controller, up to 60 degrees at full deflection and at most 180 degrees per second: pulling the stick toward you raises the far edge (positive rotation about x), pushing it right lowers the right side (negative rotation about z). The gyroscope reports the rate while the angle changes and the accelerometer reports gravity (9.80665 m/s^2) in the tilted frame;
- pressing the left stick click lifts the controller: 1 g more along the axis out of its face for 150 ms, then 1 g less for 150 ms, once per press;
- the right stick and both stick clicks do not reach the title.

Releasing the right stick click turns the controller back to level at the same rate. At rest it reports 1 g on +y and no rotation.

## Shader recompiler

The shader recompilation logic in [core/shader/recompiler](../../core/shader/recompiler) is isolated from the rest of the project and is a pure function of its input data, designed for integration into any other project. The current CMake target also includes cache support and links a supplied runtime target, glslang, and optionally SPIRV-Tools.
