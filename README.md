# Deep Learning Virtual Machine
## Core Compiler Infrastructure

Welcome to DLVM! For general information about the DLVM project,
please visit [dlvm.org](http://dlvm.org).

DLVM is:
- a framework for building DSLs
- an IR for linear algebra and neural networks
- an automatic backpropagator
- a GPU code generator (NVPTX, AMDGPU, HPVM)
- a production-quality infrastructure

## Targets

### [DLVM Core](Sources)

| Module             | Description                                             |
|--------------------|---------------------------------------------------------|
| DLVM               | Compiler infrastructure (ADT, IR, Analyses, Transforms) |
| CoreOp             | Semantics of simple tensor ops                          |
| CoreCompute        | Compute IR (stages `compute` and `schedule` of DLVM IR) |
| DLParse            | Textual IR parser                                       |
| DLVMCodeGen        | LLVM code generator                                     |
| DLCommandLineTools | Tools for building DLVM CLIs                            |
| dlopt              | CLI for DLVM optimizer                                  |
| dlc                | CLI for DLVM IR compiler                                |

### [DLVM Compute Primitives](Compute) (**deprecated**)

The DLVM compute primitives (dlcompute) are a set of LLVM Bitcode modules
containing HPVM intrinsic calls. It's built separately via CMake.

### [DLVM Runtime](Runtime)

The DLVM runtime (dlrt) includes runtime routines including memory tracking,
reference counting, etc.

## Dependencies

- LLVM 5
- Swift 4.1 dev
  - DEVELOPMENT-SNAPSHOT-2017-10-12 or above
  - **Not** 4.0-DEVELOPMENT-SNAPSHOT-*!
- Xcode Command Line Tools 9.0 (for macOS only)

## Build Instructions

### Set up LLVM (macOS)

```bash
brew install llvm
brew link --overwrite --force llvm
cp Resources/pkgconfig/llvm.pc /usr/local/lib/pkgconfig
```

### Build DLVM Core

#### Swift Package Manger

```bash
swift build
```

#### Xcode

```bash
swift package generate-xcodeproj
```

### Build All Targets

For all targets (DLVM Core, Runtime, Compute), please use CMake.

```bash
mkdir build
cd build
cmake ..
make
```

## License

[Apache 2](LICENSE)
