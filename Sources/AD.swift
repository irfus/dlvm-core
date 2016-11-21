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

extension Assignment {

    func propogateForward() {
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
            // TODO 
            break

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
                        &zero, data.descriptor.handle, dest)
                }
            }

        case let .negative(x):
            var one: DataType = -1
            var zero = DataType.zero
            self.data.withUnsafeMutableDeviceAddress { dest -> () in
                x.data.withUnsafeDeviceAddress { src in
                    !!cudnnAddTensor(
                        graph.dnn.handle,
                        &one, x.data.descriptor.handle, src,
                        &zero, data.descriptor.handle, dest
                    )
                }
            }

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
                        &one, data.descriptor.handle, dest)
                }
            }

        case let .product(lhs, rhs):
            var one: DataType = 1
            var zero = DataType.zero
            self.data.withUnsafeMutableDeviceAddress { ptrC -> () in
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

        case let .log(x):
            break

        case let .input(shape: shape):
            break

        case let .parameter(shape: shape, initial: initializer):
            break
        }

    }
}
