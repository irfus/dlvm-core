# DLVM Core
## Core Compiler Infrastructure

Welcome to DLVM! For general information about the DLVM project,
please visit [dlvm.org](http://dlvm.org).

This repository does not include a code generator.

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
- LLVM Integrated Tester
- Xcode Command Line Tools 9.0+ (for macOS only)

### Mac Installation

```sh
# Install Command Line Tools
xcode-select --install

# Install Swift via swiftenv
git clone https://github.com/kylef/swiftenv.git ~/.swiftenv
# Follow instructions to add swiftenv to path:
# https://swiftenv.fuller.li/en/latest/installation.html
swiftenv install DEVELOPMENT-SNAPSHOT-2017-12-12-a
swiftenv global DEVELOPMENT-SNAPSHOT-2017-12-12-a

# Install LLVM (for FileCheck)
brew install llvm && brew link --force --overwrite llvm

# Install LLVM Integrated Tester (lit)
pip install lit
```

### Linux Installation

```sh
# Install Swift via swiftenv
git clone https://github.com/kylef/swiftenv.git ~/.swiftenv
# Follow instructions to add swiftenv to path:
# https://swiftenv.fuller.li/en/latest/installation.html
swiftenv install DEVELOPMENT-SNAPSHOT-2017-12-12-a
swiftenv global DEVELOPMENT-SNAPSHOT-2017-12-12-a

# Install LLVM (for FileCheck)
wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-5.0 main"
sudo apt-get update
sudo apt-get install llvm-5.0

# Install LLVM Integrated Tester (lit)
sudo apt-get install python-pip
pip install lit
```

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
