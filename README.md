- Main repo: [GitHub](https://github.com/rxwei/DLVM)
- Mirror: [LLVM Group GitLab](https://gitlab-beta.engr.illinois.edu/llvm/dlvm)

# Deep Learning Virtual Machine

Welcome to DLVM! DLVM is a compiler framwork for deep learning applications. It
has an automatically differentiable IR, a set of tensor compute abstractions,
and a code generator to LLVM IR with compute functions targeting the
Heterogeneous Parallel Virtual Machine and the LLVM NVPTX backend.

## Targets

### DLVM Core

| Module      | Description                                                              |
|-------------|--------------------------------------------------------------------------|
| DLVM        | Compiler infrastructure (ADT, IR, Analyses, Transforms)                  |
| DLVMReader  | Textual IR reader                                                        |
| DLVMCodeGen | LLVM Code Generator                                                      |
| TEL, telc   | *Deprecated* TEL frontend (Moved to [TEL](https://github.com/rxwei/TEL)) |
| dlrt        | Runtime routines (reference counting, etc)                               |

### DLVM Compute Primitives

Compute primitives are a set of LLVM Bitcode modules contaning HPVM intrinsic
calls. This target is located in [/Compute](Compute). It's built separately from
DLVM Core, since the Swift Package Manager does not support LLVM Bitcode targets.

## Build Instructions

### DLVM Core

#### Swift Package Manger

```bash
swift build
```

#### Xcode

```bash
swift package generate-xcodeproj
```

### Other Targets

For all other targets, please use CMake.

```bash
mkdir build
cmake ..
make
```

## License

[Apache 2](LICENSE)
