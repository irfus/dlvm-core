*IN PROGRESS*

## Build

You can use `make`, which invokes one of the following commands depending on the
platform.

#### macOS
```
swift build -Xcc -I/usr/local/cuda/include -Xlinker -L/usr/local/cuda/lib
```

#### Linux
```
swift build -Xcc -I/usr/local/cuda/include -Xlinker -L/usr/local/cuda/lib64
```

## Components

### Core

### Next steps

- [ ] [CuDNN](https://github.com/rxwei/cudnn-swift)

You may use the Makefile in this project.

## Dependencies

- [cuda-swift](https://github.com/rxwei/CCUDA) - `CUDARuntime` only
- [CCUDA](https://github.com/rxwei/CCUDA) - `CCuDNN` module only

## License

MIT License

CUDA is a registered trademark of NVIDIA Corporation.
