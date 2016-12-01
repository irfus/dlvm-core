//
//  AD.swift
//  DLVM
//
//  Created by Richard Wei on 11/13/16.
//
//

import CUDARuntime
import CCuDNN
import CuBLAS
import Warp

extension Variable {

    /// Perform forward propagation
    func propagateForward() {
        switch rValue {
        case let .add(lhs, rhs):
            var one: DataType = 1
            var zero = 0
            self.data.withUnsafeMutableDeviceAddress { ptrC in
                lhs.data.withUnsafeDeviceAddress { ptrA in
                    rhs.data.withUnsafeDeviceAddress { ptrB in
                        !!cudnnOpTensor(
                            graph.dnn.handle,
                            graph.tensorOperators.addOp,
                            &one, lhs.data.descriptor.handle, ptrA,
                            &one, rhs.data.descriptor.handle, ptrB,
                            &zero, self.data.descriptor.handle, ptrC
                        )
                    }
                }
            }

        case let .mul(lhs, rhs):
            var one: DataType = 1
            var zero = 0
            self.data.withUnsafeMutableDeviceAddress { ptrC in
                lhs.data.withUnsafeDeviceAddress { ptrA in
                    rhs.data.withUnsafeDeviceAddress { ptrB in
                        !!cudnnOpTensor(
                            graph.dnn.handle,
                            graph.tensorOperators.mulOp,
                            &one, lhs.data.descriptor.handle, ptrA,
                            &one, rhs.data.descriptor.handle, ptrB,
                            &zero, self.data.descriptor.handle, ptrC
                        )
                    }
                }
            }

        case let .min(lhs, rhs):
            var one: DataType = 1
            var zero = 0
            self.data.withUnsafeMutableDeviceAddress { ptrC in
                lhs.data.withUnsafeDeviceAddress { ptrA in
                    rhs.data.withUnsafeDeviceAddress { ptrB in
                        !!cudnnOpTensor(
                            graph.dnn.handle,
                            graph.tensorOperators.minOp,
                            &one, lhs.data.descriptor.handle, ptrA,
                            &one, rhs.data.descriptor.handle, ptrB,
                            &zero, self.data.descriptor.handle, ptrC
                        )
                    }
                }
            }

        case let .max(lhs, rhs):
            var one: DataType = 1
            var zero = 0
            self.data.withUnsafeMutableDeviceAddress { ptrC in
                lhs.data.withUnsafeDeviceAddress { ptrA in
                    rhs.data.withUnsafeDeviceAddress { ptrB in
                        !!cudnnOpTensor(
                            graph.dnn.handle,
                            graph.tensorOperators.maxOp,
                            &one, lhs.data.descriptor.handle, ptrA,
                            &one, rhs.data.descriptor.handle, ptrB,
                            &zero, self.data.descriptor.handle, ptrC
                        )
                    }
                }
            }
            
        case let .tanh(x):
            var one: DataType = 1
            var zero = 0
            self.data.withUnsafeMutableDeviceAddress { dest in
                x.data.withUnsafeDeviceAddress { src in
                    !!cudnnActivationForward(
                        graph.dnn.handle,
                        graph.tensorOperators.tanhActivation,
                        &one, x.data.descriptor.handle, src,
                        &zero, data.descriptor.handle, dest
                    )
                }
            }

        case let .relu(x):
            var one: DataType = 1
            var zero = 0
            self.data.withUnsafeMutableDeviceAddress { dest in
                x.data.withUnsafeDeviceAddress { src in
                    !!cudnnActivationForward(
                        graph.dnn.handle,
                        graph.tensorOperators.reluActivation,
                        &one, x.data.descriptor.handle, src,
                        &zero, data.descriptor.handle, dest
                    )
                }
            }

        case let .sigmoid(x):
            var one: DataType = 1
            var zero = 0
            self.data.withUnsafeMutableDeviceAddress { dest in
                x.data.withUnsafeDeviceAddress { src in
                    !!cudnnActivationForward(
                        graph.dnn.handle,
                        graph.tensorOperators.sigmoidActivation,
                        &one, x.data.descriptor.handle, src,
                        &zero, data.descriptor.handle, dest
                    )
                }
            }
            
        case let .softmax(x):
            var one: DataType = 1
            var zero = 0
            self.data.withUnsafeMutableDeviceAddress { dest in
                x.data.withUnsafeDeviceAddress { src in
                    !!cudnnSoftmaxForward(
                        graph.dnn.handle,
                        CUDNN_SOFTMAX_LOG,
                        CUDNN_SOFTMAX_MODE_CHANNEL,
                        &one, x.data.descriptor.handle, src,
                        &zero, data.descriptor.handle, dest
                    )
                }
            }

        case let .negative(x):
            self.data.elements.assign(from: x.data.elements, multipliedBy: -1)

        case let .scalarComplement(lhs, rhs):
            var minusOne: DataType = -1
            var one: DataType = 1
            var lhs = lhs
            self.data.withUnsafeMutableDeviceAddress { dest -> () in
                rhs.data.withUnsafeDeviceAddress { src in
                    !!cudnnSetTensor(
                        graph.dnn.handle,
                        data.descriptor.handle, dest, &lhs)
                    !!cudnnTransformTensor(
                        graph.dnn.handle,
                        &minusOne, rhs.data.descriptor.handle, src,
                        &one, data.descriptor.handle, dest
                    )
                }
            }

        case let .product(lhs, rhs):
            self.data.withUnsafeMutableDevicePointer { C in
                lhs.data.withUnsafeDevicePointer { A in
                    rhs.data.withUnsafeDevicePointer { B in
                        /// Matrix multiplication
                        /// This implementation compromises static type checking,
                        /// due to cuBLAS GEMM being non-generic. Will need a better
                        /// generalization of GEMM.
                        let blas = graph.blas
                        if DataType.self == Float.self {
                            let ptrA = unsafeBitCast(A, to: UnsafeDevicePointer<Float>.self)
                            let ptrB = unsafeBitCast(B, to: UnsafeDevicePointer<Float>.self)
                            let ptrC = unsafeBitCast(C, to: UnsafeMutableDevicePointer<Float>.self)
                            blas.gemm(
                                alpha: 1.0,
                                A: ptrA, rowCount: Int32(lhs.shape[0]),
                                transpose: .none, leadingDimension: Int32(lhs.shape[0]),
                                B: ptrB, columnCount: Int32(rhs.shape.dimensions.last!),
                                transpose: .none, leadingDimension: Int32(rhs.shape[0]),
                                commonDimension: Int32(rhs.shape[0]),
                                beta: 0.0, C: ptrC, leadingDimension: Int32(self.shape[0])
                            )
                        }
                        else if DataType.self == Double.self {
                            let ptrA = unsafeBitCast(A, to: UnsafeDevicePointer<Double>.self)
                            let ptrB = unsafeBitCast(B, to: UnsafeDevicePointer<Double>.self)
                            let ptrC = unsafeBitCast(C, to: UnsafeMutableDevicePointer<Double>.self)
                            blas.gemm(
                                alpha: 1.0,
                                A: ptrA, rowCount: Int32(lhs.shape[0]),
                                transpose: .none, leadingDimension: Int32(lhs.shape[0]),
                                B: ptrB, columnCount: Int32(rhs.shape.dimensions.last!),
                                transpose: .none, leadingDimension: Int32(rhs.shape[0]),
                                commonDimension: Int32(rhs.shape[0]),
                                beta: 0.0, C: ptrC, leadingDimension: Int32(self.shape[0])
                            )
                        }
                        else {
                            fatalError("Data type not supported by cuBLAS GEMM")
                        }
                    }
                }
            }

        case let .log(x):
            self.data.elements.assign(from: x.data.elements, transformedBy: .log)

        case .input:
            break

        case .parameter:
            break
        }
    }

    /// Perform forward propagation
    func propagateBackward() {
        switch rValue {
        case let .add(lhs, rhs):
            var one: DataType = 1
            var zero = 0
            self.data.withUnsafeMutableDeviceAddress { ptrC in
                lhs.data.withUnsafeDeviceAddress { ptrA in
                    rhs.data.withUnsafeDeviceAddress { ptrB in
                        /// TODO: derivative
                    }
                }
            }

        case let .mul(lhs, rhs):
            var one: DataType = 1
            var zero = 0
            self.data.withUnsafeMutableDeviceAddress { ptrC in
                lhs.data.withUnsafeDeviceAddress { ptrA in
                    rhs.data.withUnsafeDeviceAddress { ptrB in
                        /// TODO: derivative
                    }
                }
            }

        case let .min(lhs, rhs):
            var one: DataType = 1
            var zero = 0
            self.data.withUnsafeMutableDeviceAddress { ptrC in
                lhs.data.withUnsafeDeviceAddress { ptrA in
                    rhs.data.withUnsafeDeviceAddress { ptrB in
                        /// TODO: derivative
                    }
                }
            }

        case let .max(lhs, rhs):
            var one: DataType = 1
            var zero = 0
            self.data.withUnsafeMutableDeviceAddress { ptrC in
                lhs.data.withUnsafeDeviceAddress { ptrA in
                    rhs.data.withUnsafeDeviceAddress { ptrB in
                        /// TODO: derivative
                    }
                }
            }
            
        case let .tanh(arg):
            var one: DataType = 1
            var zero = 0
            self.data.withUnsafeDeviceAddress { y in
                self.gradient.withUnsafeDeviceAddress { dy in
                    arg.data.withUnsafeDeviceAddress { x in
                        arg.gradient.withUnsafeMutableDeviceAddress { dx in
                            !!cudnnActivationBackward(
                                graph.dnn.handle,
                                graph.tensorOperators.tanhActivation,
                                &one,
                                self.data.descriptor.handle, y,
                                self.gradient.descriptor.handle, dy,
                                arg.data.descriptor.handle, x, &zero,
                                arg.gradient.descriptor.handle, dx
                            )
                        }
                    }
                }
            }

        case let .relu(arg):
            var one: DataType = 1
            var zero = 0
            self.data.withUnsafeDeviceAddress { y in
                self.gradient.withUnsafeDeviceAddress { dy in
                    arg.data.withUnsafeDeviceAddress { x in
                        arg.gradient.withUnsafeMutableDeviceAddress { dx in
                            !!cudnnActivationBackward(
                                graph.dnn.handle,
                                graph.tensorOperators.reluActivation,
                                &one,
                                self.data.descriptor.handle, y,
                                self.gradient.descriptor.handle, dy,
                                arg.data.descriptor.handle, x, &zero,
                                arg.gradient.descriptor.handle, dx
                            )
                        }
                    }
                }
            }

        case let .sigmoid(arg):
            var one: DataType = 1
            var zero = 0
            self.data.withUnsafeDeviceAddress { y in
                self.gradient.withUnsafeDeviceAddress { dy in
                    arg.data.withUnsafeDeviceAddress { x in
                        arg.gradient.withUnsafeMutableDeviceAddress { dx in
                            !!cudnnActivationBackward(
                                graph.dnn.handle,
                                graph.tensorOperators.sigmoidActivation,
                                &one,
                                self.data.descriptor.handle, y,
                                self.gradient.descriptor.handle, dy,
                                arg.data.descriptor.handle, x, &zero,
                                arg.gradient.descriptor.handle, dx
                            )
                        }
                    }
                }
            }

        case let .softmax(arg):
            var one: DataType = 1
            var zero = 0
            self.data.withUnsafeDeviceAddress { y in
                self.gradient.withUnsafeDeviceAddress { dy in
                    arg.gradient.withUnsafeMutableDeviceAddress { dx in
                        !!cudnnSoftmaxBackward(
                            graph.dnn.handle,
                            CUDNN_SOFTMAX_LOG,
                            CUDNN_SOFTMAX_MODE_CHANNEL,
                            &one,
                            self.data.descriptor.handle, y,
                            self.gradient.descriptor.handle, dy,
                            &zero, arg.gradient.descriptor.handle, dx
                        )
                    }
                }
            }

        case let .negative(x):
            self.data.elements.assign(from: x.data.elements, multipliedBy: -1)

        case let .scalarComplement(lhs, rhs):
            var minusOne: DataType = -1
            var one: DataType = 1
            var lhs = lhs
            self.data.withUnsafeMutableDeviceAddress { dest -> () in
                rhs.data.withUnsafeDeviceAddress { src in
                    /// TODO: derivative
                }
            }

        case let .product(lhs, rhs):
            self.data.withUnsafeMutableDevicePointer { C in
                lhs.data.withUnsafeDevicePointer { A in
                    rhs.data.withUnsafeDevicePointer { B in
                        /// Matrix multiplication
                        /// This implementation compromises static type checking,
                        /// due to cuBLAS GEMM being non-generic. Will need a better
                        /// generalization of GEMM.
                        let blas = graph.blas
                        if DataType.self == Float.self {
                            let ptrA = unsafeBitCast(A, to: UnsafeDevicePointer<Float>.self)
                            let ptrB = unsafeBitCast(B, to: UnsafeDevicePointer<Float>.self)
                            let ptrC = unsafeBitCast(C, to: UnsafeMutableDevicePointer<Float>.self)
                            /// TODO
                        }
                        else if DataType.self == Double.self {
                            let ptrA = unsafeBitCast(A, to: UnsafeDevicePointer<Double>.self)
                            let ptrB = unsafeBitCast(B, to: UnsafeDevicePointer<Double>.self)
                            let ptrC = unsafeBitCast(C, to: UnsafeMutableDevicePointer<Double>.self)
                            /// TODO
                        }
                        else {
                            fatalError("Data type not supported by cuBLAS GEMM")
                        }
                    }
                }
            }

        case let .log(x):
            /// TODO: derivative
            break

        case .input:
            break

        case .parameter:
            break
        }
    }
    
}
