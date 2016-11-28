*IN PROGRESS*

This is under **very active** development. Components are **not** ready to use. 

# The DLVM Infrastructure for Neural Networks

- [x] Embedded DSL in Swift
- [ ] TEL - The Tensor Expression Language
    - [ ] Parser
    - [ ] Tensor type checker
    - [x] IR generator
- [ ] DLVM IR - Intermediate Representation
    - [x] Assignment form
    - [ ] Φ function
    - [ ] Loops
    - [ ] Optimization passes
- [ ] Automatic Differentiation
    - [ ] GPU Interpreter
        - [x] Forward propagation
        - [ ] Backward propagation
    - [ ] GPU (PTX) CodeGen
    - [ ] CPU CodeGen

## Stages

1. CUDA Backend

2. HPVM Backend

3. Optimization

## Embedded DSL in Swift

Syntax:

```swift
let x = Expression<Float>.input(shape: [2, 1], name: "x")
let W1 = Expression<Float>.parameter(shape: [2, 2],
                                     initial: .random(from: 0.0, to: 1.0), 
                                     name: "W1")
let b1 = Expression<Float>.parameter(shape: [2, 1], 
                                     initial: .random(from: 0.0, to: 1.0),
                                     name: "b1")
let W2 = Expression<Float>.parameter(shape: [4, 2],
                                     initial: .zeros, 
                                     name: "W2")
let b2 = Expression<Float>.parameter(shape: [4, 1], 
                                     initial: .zeros, 
                                     name: "b2")
let h1 = tanh(W1 • x + b1) ~ "h1"          /// Hidden layer 1 named "h1" 
let o = softmax(W2 • (1 - h1) + b2) /// Output layer

/// Interpret the expression and build the computable graph (DLVM IR)
let graph = try Graph<Float>(expression: o)
``````

## TEL

Proposed syntax:

```
[type: Float]
[output: o]

x = input { shape: [2x1] }

W1 = parameter { shape: [2x2], initial: random(0.0, 1.0) } 
b1 = parameter { shape: [2x1], initial: 0.0 }
W2 = parameter { shape: [4x2], initial: random(0.0, 1.0) }
b2 = parameter { shape: [4x2], initial: 0.0 }

h1 = tanh(W1 x + b1)
o = softmax(W2 (1 - h1) + b2)
``````

## DLVM IR

Proposed syntax:

```
label:
    float [4x2] W2 = param:[4x2]
    float [2x2] W1 = param:[2x2]
    float [2x1] x = input:[2x1]
    float [2x1] v1 = W1 • x
    float [2x1] b1 = param:[2x1]
    float [2x1] v2 = v1 + b1
    float [2x1] h1 = tanh(v)
    float [2x1] v4 = 1.0 - h1
    float [4x1] v5 = W2 • v4
    float [4x1] b2 = param:[4x1]
    float [4x1] v6 = v5 + b2
    float [4x1] h2 = softmax(v6)
``````

Currently DLVM IR does not support loops. It's a goal for the second stage, 
whereas loops are necessary for recurrent neural networks.

## Build

You can use `make`, which invokes one of the following commands depending on the
platform.

### macOS
```
swift build -Xcc -I/usr/local/cuda/include -Xlinker -L/usr/local/cuda/lib
``````

### Linux
```
swift build -Xcc -I/usr/local/cuda/include -Xlinker -L/usr/local/cuda/lib64
``````

## Dependencies

- [cuda-swift](https://github.com/rxwei/CCUDA) - `CUDARuntime`, `CuBLAS`
- [CCUDA](https://github.com/rxwei/CCUDA) - `CCuDNN`
- NVIDIA CUDA library

## License

MIT License

CUDA is a registered trademark of NVIDIA Corporation.

