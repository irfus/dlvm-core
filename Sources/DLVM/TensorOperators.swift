//
//  TensorOperators.swift
//  LLNM
//
//  Created by Richard Wei on 11/20/16.
//
//

import CCuDNN

class TensorOperators<DataType: TensorDataProtocol> {

    fileprivate let opDescriptor: cudnnOpTensorDescriptor_t = {
        var opDescriptor: cudnnOpTensorDescriptor_t?
        !!cudnnCreateOpTensorDescriptor(&opDescriptor)
        return opDescriptor!
    }()

    fileprivate let activationDescriptor: cudnnActivationDescriptor_t = {
        var activationDescriptor: cudnnActivationDescriptor_t?
        !!cudnnCreateActivationDescriptor(&activationDescriptor)
        return activationDescriptor!
    }()

    deinit {
        cudnnDestroyOpTensorDescriptor(opDescriptor)
        cudnnDestroyActivationDescriptor(activationDescriptor)
    }

}

extension TensorOperators {
    
    var addOp: cudnnOpTensorDescriptor_t {
        !!cudnnSetOpTensorDescriptor(
            opDescriptor,
            CUDNN_OP_TENSOR_ADD,
            DataType.tensorDataType.cType,
            CUDNN_PROPAGATE_NAN
        )
        return opDescriptor
    }
    
    var mulOp: cudnnOpTensorDescriptor_t {
        !!cudnnSetOpTensorDescriptor(
            opDescriptor,
            CUDNN_OP_TENSOR_MUL,
            DataType.tensorDataType.cType,
            CUDNN_PROPAGATE_NAN
        )
        return opDescriptor
    }
    
    var minOp: cudnnOpTensorDescriptor_t {
        !!cudnnSetOpTensorDescriptor(
            opDescriptor,
            CUDNN_OP_TENSOR_MIN,
            DataType.tensorDataType.cType,
            CUDNN_PROPAGATE_NAN
        )
        return opDescriptor
    }
    
    var maxOp: cudnnOpTensorDescriptor_t {
        !!cudnnSetOpTensorDescriptor(
            opDescriptor,
            CUDNN_OP_TENSOR_MAX,
            DataType.tensorDataType.cType,
            CUDNN_PROPAGATE_NAN
        )
        return opDescriptor
    }
    
    var sigmoidActivation: cudnnActivationDescriptor_t {
        !!cudnnSetActivationDescriptor(
            activationDescriptor,
            CUDNN_ACTIVATION_SIGMOID,
            CUDNN_PROPAGATE_NAN,
            1.0
        )
        return activationDescriptor
    }
    
    var tanhActivation: cudnnActivationDescriptor_t {
        !!cudnnSetActivationDescriptor(
            activationDescriptor,
            CUDNN_ACTIVATION_TANH,
            CUDNN_PROPAGATE_NAN,
            1.0
        )
        return activationDescriptor
    }
    
    var reluActivation: cudnnActivationDescriptor_t {
        !!cudnnSetActivationDescriptor(
            activationDescriptor,
            CUDNN_ACTIVATION_RELU,
            CUDNN_PROPAGATE_NAN,
            1.0
        )
        return activationDescriptor
    }
    
}
