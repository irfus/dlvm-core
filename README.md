*IN PROGRESS*

This is under **very active** development. Components are **not** ready to use. 

# The DLVM Infrastructure for Neural Networks

- [ ] TEL - The Tensor Expression Language
    - [ ] Parse
    - [ ] Sema
    - [x] DLVM IR CodeGen
- [x] DLVM IR - The Intermediate Representation for Computation
    - [x] Assignment form
    - [ ] Φ function
    - [ ] Loops
    - [ ] Optimization passes
- [x] Automatic Differentiation
    - [x] GPU Interpreter
    - [ ] GPU (PTX) CodeGen
    - [ ] CPU CodeGen

## Build

You can use `make`, which invokes one of the following commands depending on the
platform.

### macOS
```
swift build -Xcc -I/usr/local/cuda/include -Xlinker -L/usr/local/cuda/lib
```

### Linux
```
swift build -Xcc -I/usr/local/cuda/include -Xlinker -L/usr/local/cuda/lib64
```

## Dependencies

- [cuda-swift](https://github.com/rxwei/CCUDA) - `CUDARuntime` only
- [CCUDA](https://github.com/rxwei/CCUDA) - `CCuDNN` module only

## License

MIT License

CUDA is a registered trademark of NVIDIA Corporation.
