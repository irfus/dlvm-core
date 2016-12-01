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
            !!cudnnOpTensor(
                graph.dnn.handle,
                graph.tensorOperators.addOp,
                &one, lhs.data.descriptor.handle, .&lhs.data,
                &one, rhs.data.descriptor.handle, .&rhs.data,
                &zero, self.data.descriptor.handle, !&self.data
            )

        case let .mul(lhs, rhs):
            var one: DataType = 1
            var zero = 0
            !!cudnnOpTensor(
                graph.dnn.handle,
                graph.tensorOperators.mulOp,
                &one, lhs.data.descriptor.handle, .&lhs.data,
                &one, rhs.data.descriptor.handle, .&rhs.data,
                &zero, self.data.descriptor.handle, !&self.data
            )

        case let .min(lhs, rhs):
            var one: DataType = 1
            var zero = 0
            !!cudnnOpTensor(
                graph.dnn.handle,
                graph.tensorOperators.minOp,
                &one, lhs.data.descriptor.handle, .&lhs.data,
                &one, rhs.data.descriptor.handle, .&rhs.data,
                &zero, self.data.descriptor.handle, !&self.data
            )

        case let .max(lhs, rhs):
            var one: DataType = 1
            var zero = 0
            !!cudnnOpTensor(
                graph.dnn.handle,
                graph.tensorOperators.maxOp,
                &one, lhs.data.descriptor.handle, .&lhs.data,
                &one, rhs.data.descriptor.handle, .&rhs.data,
                &zero, self.data.descriptor.handle, !&self.data
            )

        case let .tanh(x):
            var one: DataType = 1
            var zero = 0
            !!cudnnActivationForward(
                graph.dnn.handle,
                graph.tensorOperators.tanhActivation,
                &one, x.data.descriptor.handle, .&x.data,
                &zero, data.descriptor.handle, !&self.data
            )

        case let .relu(x):
            var one: DataType = 1
            var zero = 0
            !!cudnnActivationForward(
                graph.dnn.handle,
                graph.tensorOperators.reluActivation,
                &one, x.data.descriptor.handle, .&x.data,
                &zero, data.descriptor.handle, !&self.data
            )

        case let .sigmoid(x):
            var one: DataType = 1
            var zero = 0
            !!cudnnActivationForward(
                graph.dnn.handle,
                graph.tensorOperators.sigmoidActivation,
                &one, x.data.descriptor.handle, .&x.data,
                &zero, data.descriptor.handle, !&self.data
            )

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
            self.data.elements.assign(from: .multiplication, left: lhs, right: rhs.data.elements)
            
        case let .product(lhs, rhs):
            /// Matrix multiplication
            /// This implementation compromises static type checking,
            /// due to cuBLAS GEMM being non-generic. Will need a better
            /// generalization of GEMM.
            let blas = graph.blas
            if DataType.self == Float.self {
                let ptrA = unsafeBitCast(.&lhs.data, to: UnsafeDevicePointer<Float>.self)
                let ptrB = unsafeBitCast(.&rhs.data, to: UnsafeDevicePointer<Float>.self)
                let ptrC = unsafeBitCast(!&self.data, to: UnsafeMutableDevicePointer<Float>.self)
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
                let ptrA = unsafeBitCast(.&lhs.data, to: UnsafeDevicePointer<Double>.self)
                let ptrB = unsafeBitCast(.&rhs.data, to: UnsafeDevicePointer<Double>.self)
                let ptrC = unsafeBitCast(!&self.data, to: UnsafeMutableDevicePointer<Double>.self)
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
            !!cudnnActivationBackward(
                graph.dnn.handle,
                graph.tensorOperators.tanhActivation,
                &one,
                self.data.descriptor.handle, .&self.data,
                self.gradient.descriptor.handle, .&self.gradient,
                arg.data.descriptor.handle, .&arg.data, &zero,
                arg.gradient.descriptor.handle, !&arg.gradient
            )

        case let .relu(arg):
            var one: DataType = 1
            var zero = 0
            !!cudnnActivationBackward(
                graph.dnn.handle,
                graph.tensorOperators.reluActivation,
                &one,
                self.data.descriptor.handle, .&self.data,
                self.gradient.descriptor.handle, .&self.gradient,
                arg.data.descriptor.handle, .&arg.data, &zero,
                arg.gradient.descriptor.handle, !&arg.gradient
            )

        case let .sigmoid(arg):
            var one: DataType = 1
            var zero = 0
            !!cudnnActivationBackward(
                graph.dnn.handle,
                graph.tensorOperators.sigmoidActivation,
                &one,
                self.data.descriptor.handle, .&self.data,
                self.gradient.descriptor.handle, .&self.gradient,
                arg.data.descriptor.handle, .&arg.data, &zero,
                arg.gradient.descriptor.handle, !&arg.gradient
            )
            
        case let .softmax(arg):
            var one: DataType = 1
            var zero = 0
            !!cudnnSoftmaxBackward(
                graph.dnn.handle,
                CUDNN_SOFTMAX_LOG,
                CUDNN_SOFTMAX_MODE_CHANNEL,
                &one,
                self.data.descriptor.handle, .&self.data,
                self.gradient.descriptor.handle, .&self.gradient,
                &zero, arg.gradient.descriptor.handle, !&arg.gradient
            )

        case let .negative(x):
            self.data.elements.assign(from: x.data.elements, multipliedBy: -1)

        case let .scalarComplement(lhs, rhs):
            break

        case let .product(lhs, rhs):
            break
            
        case let .log(arg):
            /// dy/dx = 1/x
            let dy = self.gradient.elements
            let y = self.data.elements
            var dx = arg.gradient.elements
            dx.assign(from: .division, left: 1.0, right: y)
            dx.formElementwise(.multiplication, with: dy)
            break

        case .input:
            break

        case .parameter:
            break
        }
    }
    
}

prefix operator .&
prefix operator !&

extension DeviceTensor {

    public static prefix func .& (tensor: DeviceTensor<Element>) -> UnsafePointer<Element> {
        return tensor.elements.unsafeDevicePointer.deviceAddress
    }

    public static prefix func !& (tensor: inout DeviceTensor<Element>) -> UnsafeMutablePointer<Element> {
        return tensor.elements.unsafeMutableDevicePointer.deviceAddress
    }
    
}
