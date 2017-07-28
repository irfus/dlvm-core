- Main repo: [GitHub](https://github.com/rxwei/DLVM)
- Mirror: [LLVM Group GitLab](https://gitlab-beta.engr.illinois.edu/llvm/dlvm)

# Deep Learning Virtual Machine

Welcome to DLVM! DLVM is:
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
| DLParse            | Textual IR parser                                       |
| DLVMCodeGen        | LLVM Code Generator                                     |
| DLCommandLineTools | Tools for building DLVM CLIs                            |
| dlopt              | CLI for DLVM optimizer                                  |
| dlc                | CLI for DLVM IR compiler                                |

### [DLVM Compute Primitives](Compute)

The DLVM compute primitives (dlcompute) are a set of LLVM Bitcode modules
containing HPVM intrinsic calls. It's built separately via CMake.

### [DLVM Runtime](Runtime)

The DLVM runtime (dlrt) includes runtime routines including memory tracking,
reference counting, etc.

## Dependencies

- LLVM 4.0
- Swift 4.0 (Development Snapshot 2017-07-27 or above)

## Build Instructions

### All Targets

For all targets (DLVM Core, Runtime, Compute), please use CMake.

```bash
mkdir build
cmake ..
make
```

### DLVM Core

#### Swift Package Manger

```bash
swift build
```

#### Xcode

```bash
swift package generate-xcodeproj
```

## License

[Apache 2](LICENSE)
