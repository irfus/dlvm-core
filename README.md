Under **very active** development. Components are **not** ready to use.

- Main repo: [GitHub](https://github.com/rxwei/DLVM)
- Mirror: [LLVM Group GitLab](https://gitlab-beta.engr.illinois.edu/llvm/dlvm)

# Deep Learning Virtual Machine
## Compiler Infrastructure for Deep Neural Networks

- [ ] libDLVM
    - [ ] IR
      - [ ] Parser
      - [ ] Sema
    - [ ] BasicBlock
    - [x] Instruction
    - [x] Intrinsic
    - [ ] IRBuilder
    - [ ] BPGen (a transformation for backpropagation using automatic differentiation)
    - [ ] OptGen (a transformation for NN optimization generation, e.g. SGD)
    - [ ] IRGen (HPVM/LLVM IR)
    - [ ] ExecutionEngine
- [ ] libTEL - The Tensor Expression Language
    - [x] Parser
    - [ ] Sema
    - [ ] DLGen (DLVM IR)
    - [ ] Special Networks
        - [ ] RNN
        - [ ] CNN
        - [ ] GRU
        - [ ] LSTM
- [ ] DLVM Toolchain
    - [ ] telc - TEL compiler driver
    - [ ] dlc
- [ ] Swift-TEL Bridge
- [ ] Swift Runtime Library
    - [ ] Trainer
    - [ ] Batch management
    - [ ] Stochastic optimizer
    - [ ] Execution control
- [ ] Interpretation Engines
    - [ ] CPU
        - [ ] Forward propagation
        - [ ] Backward propagation
    - [ ] CUDA GPU
        - [ ] Forward propagation
        - [ ] Backward propagation
    - [ ] Metal GPU
        - [ ] Forward propagation
        - [ ] Backward propagation

## TEL AST builder (embedded DSL)

Syntax:

```swift
let x = Expression<Float>.input(shape: [2, 1]) ~ "x"
let W1 = Expression<Float>.parameter(shape: [2, 2],
                                     initial: .random(from: 0.0, to: 1.0)) ~ "W1"
let b1 = Expression<Float>.parameter(shape: [2, 1], 
                                     initial: .random(from: 0.0, to: 1.0)) ~ "b1"
let W2 = Expression<Float>.parameter(shape: [4, 2], initial: .zeros) ~ "W2"
let b2 = Expression<Float>.parameter(shape: [4, 1], initial: .zeros) ~ "b2"
let h1 = tanh(W1 • x + b1) ~ "h1"         /// Hidden layer 1 named "h1" 
let o = softmax(W2 • (1 - h1) + b2) ~ "o" /// Output layer named "o"

/// Interpret the expression and build the computable graph (DLVM IR)
let graph = try Graph<Float>(expression: o)
``````

## TEL

Proposed syntax:

```
[type: float16]

x: in[2x1]

W1: param[auto] = 0.0
h1: layer[4x1] = tanh(W1 x + b1)

recurrent t {
    W2, W3, W4, W5: param[auto] = random(from: 0.0, to: 1.0)
    b2, b3, b4, b5: param[auto] = 0.0
    h2: layer[16x1] = tanh(W2 [h1.t, h5.(t-1)] + b2)
    h3: layer[128x1] = tanh(W3 h2 + b3)
    h4: layer[128x1] = relu(W4 h3 + b4)
    h5: layer[16x1] = tanh(W5 h4 + b5)
}

W6: param[auto] = random(from: 0.5, to: 1.0)
b6: param[auto] = 0.0
h6: layer[16x1] = sigmoid(W6 h5 + b6)

W7: param[auto] = random(from: 0.0, to: 1.0)
b7: param[auto] = 0.0
o: out[16x1] = softmax(W7 h6 + b7)
``````

## DLVM IR

Proposed syntax:

```
entry:
    [f32 4x2] W2 = param:[4x2]
    [f32 2x2] W1 = param:[2x2]
    [f32 2x1] x = input:[2x1]
    [f32 2x1] b1 = param:[2x1]
    [f32 4x1] b2 = param:[4x1]
    int c0 = 0 /// RNN recurrence counter
label1:
    [f32 2x1] x1 = phi x, h2
    int c1 = phi c0, c2
    [f32 2x1] v1 = matmul W1, x1
    [f32 2x1] v2 = add v1, b1
    [f32 2x1] h1 = tanh v
    [f32 2x1] v4 = sub float '[1.0], h1
    [f32 4x1] v5 = matmul W2, v4
    [f32 4x1] v6 = add v5, b2
    [f32 4x1] h2 = softmax v6
    bool v7 = cmp.< c0, int 20
    branch v7 ? label1 : label2
label2:
    ...
``````

Currently DLVM IR does not support loops. It's a goal for the second stage, 
whereas loops are necessary for recurrent neural networks.

## Build

You can use `swift build`, which invokes one of the following commands depending on the
platform.

## Dependencies

- Parsey

## License

MIT License

CUDA is a registered trademark of NVIDIA Corporation.

