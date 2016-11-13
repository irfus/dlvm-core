//
//  Tensor.swift
//  Tensor
//
//  Created by Richard Wei on 9/28/16.
//
//

import CCuDNN
import Warp

/// cuDNN tensor descriptor
/// - parameter Element: type of elements of the tensor (Float, Double, ...)
final class TensorDescriptor<Element : TensorDataProtocol> {

    let handle: cudnnTensorDescriptor_t

    /// For now, we assume that strides are always 1.
    let strides: [Int32]

    let rankRequested: Int

    init(shape: TensorShape) {
        strides = Array(repeating: 1, count: shape.rank)
        var desc: cudnnTensorDescriptor_t?
        !!cudnnCreateTensorDescriptor(&desc)
        handle = desc!
        rankRequested = shape.rank
        let shapeComponents = shape.dimensions.map{Int32($0)}
        shapeComponents.withUnsafeBufferPointer { componentsBuf in
            strides.withUnsafeBufferPointer { stridesBuf in
                !!cudnnSetTensorNdDescriptor(
                    handle,
                    Element.tensorDataType.cType,
                    Int32(shape.rank),
                    componentsBuf.baseAddress,
                    stridesBuf.baseAddress
                )
            }
        }
    }

    deinit {
        !!cudnnDestroyTensorDescriptor(handle)
    }

    var shape: TensorShape {
        var rank: Int32 = 0
        var dataType = cudnnDataType_t(0)
        var dimensions = Array<Int32>(repeating: 1, count: rankRequested)
        var strides = dimensions
        !!cudnnGetTensorNdDescriptor(
            handle,
            Int32(rankRequested),
            &dataType,
            &rank,
            &dimensions,
            &strides
        )
        return TensorShape(dimensions: dimensions.map{Int($0)})
    }

}

/// Tensor
/// - parameter Element: type of elements of the tensor (Float, Double, ...)
public struct DeviceTensor<Element : TensorDataProtocol> {

    let descriptor: TensorDescriptor<Element>

    public let shape: TensorShape
    
    private var storage: DeviceArray<Element>

    /// Initialize from an array on GPU device.
    /// - parameter shape: shape of the tensor
    /// - parameter storage: array on device
    public init(shape: TensorShape, storage: DeviceArray<Element>) {
        self.descriptor = TensorDescriptor(shape: shape)
        self.storage = storage
        self.shape = shape
    }

    /// Allocate and initialize a tensor to a repeated value.
    /// - parameter shape: tensor shape
    /// - parameter repeating: repeated value
    public init(shape: TensorShape, repeating repeatedValue: Element) {
        descriptor = TensorDescriptor(shape: shape)
        storage = DeviceArray(repeating: repeatedValue, count: shape.contiguousSize)
        self.shape = shape
    }

    /// Allocate and initialize a tensor.
    /// - parameter shape: tensor shape
    public init(shape: TensorShape) {
        descriptor = TensorDescriptor(shape: shape)
        storage = DeviceArray(capacity: shape.contiguousSize)
        self.shape = shape
    }

    /// Compute the contiguous storage index from high-dimensional tensor indices.
    /// - parameter indices: tensor indices
    /// - returns: index in contiguous storage
    /// - note: the count of indices must equal the rank of the tensor
    private func storageIndex(from indices: [Int]) -> Int {
        /// Row-major order addressing
        return indices.enumerated().reduce(0, { acc, next in
            next.element * (next.offset..<shape.rank).reduce(0, { $0 + shape[$1] })
        })
    }

    /// Access an element of the tensor.
    /// - parameter indices: tensor indices
    /// - returns: reference to the value on GPU device
    /// - note: the count of indices must equal the rank of the tensor
    public subscript(indices: Int...) -> DeviceValue<Element> {
        get {
            guard indices.count == shape.rank else {
                fatalError("Incorrect index dimension")
            }
            return storage[storageIndex(from: indices)]
        }
        set {
            guard indices.count == shape.rank else {
                fatalError("Incorrect index dimension")
            }
            storage[storageIndex(from: indices)] = newValue
        }
    }

}

public typealias Tensor<Element : TensorDataProtocol> = DeviceTensor<Element>
