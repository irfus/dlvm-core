//
//  AD.swift
//  LLNM
//
//  Created by Richard Wei on 11/13/16.
//
//

import CUDARuntime
import CCuDNN
import CuBLAS

extension Variable {

    func propagateForward() {
        switch rValue {
        case let .add(lhs, rhs):
            var one: DataType = 1
            var zero = DataType.zero
            self.data.withUnsafeMutableDeviceAddress { ptrC -> () in
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
            self.data.elements.assignElementwiseResult(of: .multiplication,
                                                       x: lhs.data.elements,
                                                       y: rhs.data.elements)

        case let .min(lhs, rhs):
            var one: DataType = 1
            var zero = DataType.zero
            self.data.withUnsafeMutableDeviceAddress { ptrC -> () in
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
            var zero = DataType.zero
            self.data.withUnsafeMutableDeviceAddress { ptrC -> () in
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
            var zero = DataType.zero
            self.data.withUnsafeMutableDeviceAddress { dest -> () in
                x.data.withUnsafeDeviceAddress { src in
                    !!cudnnActivationForward_v4(
                        graph.dnn.handle,
                        graph.tensorOperators.tanhActivation,
                        &one, x.data.descriptor.handle, src,
                        &zero, data.descriptor.handle, dest
                    )
                }
            }

        case let .relu(x):
            var one: DataType = 1
            var zero = DataType.zero
            self.data.withUnsafeMutableDeviceAddress { dest -> () in
                x.data.withUnsafeDeviceAddress { src in
                    cudnnActivationForward_v4(
                        graph.dnn.handle,
                        graph.tensorOperators.reluActivation,
                        &one, x.data.descriptor.handle, src,
                        &zero, data.descriptor.handle, dest
                    )
                }
            }

        case let .sigmoid(x):
            var one: DataType = 1
            var zero = DataType.zero
            self.data.withUnsafeMutableDeviceAddress { dest -> () in
                x.data.withUnsafeDeviceAddress { src in
                    cudnnActivationForward_v4(
                        graph.dnn.handle,
                        graph.tensorOperators.sigmoidActivation,
                        &one, x.data.descriptor.handle, src,
                        &zero, data.descriptor.handle, dest
                    )
                }
            }
            
        case let .softmax(x):
            var one: DataType = 1
            var zero = DataType.zero
            self.data.withUnsafeMutableDeviceAddress { dest -> () in
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
            self.data.elements.assign(x.data.elements, multipliedBy: -1)

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
            let blas = graph.blas
            self.data.elements.withUnsafeMutableDevicePointer { ptrC in
                lhs.data.elements.withUnsafeDevicePointer { ptrA in
                    rhs.data.elements.withUnsafeDevicePointer { ptrB in
                        blas.gemm(
                            alpha: 1.0,
                            A: ptrA, rowCount: Int32(lhs.shape.dimensions.first!),
                            transpose: .none, leadingDimension: Int32(lhs.shape.leadingDimension),
                            B: ptrB, columnCount: Int32(rhs.shape.leadingDimension),
                            transpose: .none, leadingDimension: Int32(rhs.shape.leadingDimension),
                            commonDimension: Int32(lhs.shape.dimensions.last!),
                            beta: 0.0, C: ptrC, leadingDimension: Int32(self.shape.leadingDimension)
                        )
                    }
                }
            }

        case let .log(x):
            self.data.elements.assign(x.data.elements, transformedBy: .log)

        case .input:
            break

        case .parameter:
            break
        }

    }
}
