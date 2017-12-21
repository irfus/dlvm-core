# Deep Learning Virtual Machine
## Core Compiler Infrastructure

Welcome to DLVM! For general information about the DLVM project,
please visit [dlvm.org](http://dlvm.org).

DLVM is:
- a framework for building DSLs
- an IR for linear algebra and neural networks
- an automatic backpropagator
- a production-quality infrastructure

## Targets

### [DLVM Core](Sources)

| Module             | Description                                             |
|--------------------|---------------------------------------------------------|
| DLVM               | Compiler infrastructure (ADT, IR, Analyses, Transforms) |
| CoreOp             | Semantics of simple tensor ops                          |
| DLParse            | Textual IR parser                                       |
| DLCommandLineTools | Tools for building DLVM CLIs                            |
| dlopt              | CLI for DLVM optimizer                                  |

### [DLVM Runtime](Runtime)

The DLVM runtime (dlrt) includes runtime routines including memory tracking,
reference counting, etc.

## Dependencies

- Swift 4.1 dev
  - DEVELOPMENT-SNAPSHOT-2017-12-11 or above
- LLVM
  - `brew install llvm && brew link --force --overwrite llvm`
- LLVM Integrated Tester
  - `pip install lit`
- Xcode Command Line Tools 9.0+ (for macOS only)

## Build Instructions

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

For all targets (DLVM Core and Runtime), please use CMake.

```bash
mkdir build
cd build
cmake ..
make
```

## License

[Apache 2](LICENSE)
