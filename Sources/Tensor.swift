//
//  Tensor.swift
//  Tensor
//
//  Created by Richard Wei on 9/28/16.
//
//

import CCuDNN
import Warp

/// Tensor shape
public struct TensorShape : ExpressibleByArrayLiteral {

    public var dimensions: [Int]

    var strides: [Int]

    /// Initialize with rank, and set the size of each dimension to 1.
    /// - parameter rank: rank of the tensor
    public init(rank: Int) {
        dimensions = Array(repeating: 1, count: rank)
        strides = dimensions
    }

    /// Initialize with sizes of dimensions. The rank of the tensor
    /// is the length of the parameter list.
    /// - parameter dimensions: sizes of dimensions
    public init(dimensions: [Int]) {
        self.dimensions = dimensions
        strides = Array(repeating: 1, count: dimensions.count)
    }

    /// Initialize with sizes of dimensions. The rank of the tensor
    /// is the length of the parameter list.
    /// - parameter dimensions: sizes of dimensions
    public init(_ dimensions: Int...) {
        self.dimensions = dimensions
        strides = Array(repeating: 1, count: dimensions.count)
    }

    /// Initialize with an array literal, representing the sizes of
    /// dimensions. The rank of the tensor is the length of the parameter
    /// list.
    /// - parameter dimensions: sizes of dimensions
    public init(arrayLiteral elements: Int...) {
        self.dimensions = elements
        strides = Array(repeating: 1, count: elements.count)
    }

    /// Get the size of i-th dimension.
    /// - parameter i: dimension
    public subscript(i: Int) -> Int {
        return dimensions[i]
    }

    /// Rank of the tensor
    public var rank: Int {
        return dimensions.count
    }

    /// Size of the tensor as a contiguously stored array
    public var contiguousSize: Int {
        return dimensions.reduce(1, *)
    }

}

/// cuDNN tensor descriptor
/// - parameter Element: type of elements of the tensor (Float, Double, ...)
final class TensorDescriptor<Element : TensorDataProtocol> {

    let handle: cudnnTensorDescriptor_t

    var rankRequested: Int

    init(shape: TensorShape) {
        var desc: cudnnTensorDescriptor_t?
        !!cudnnCreateTensorDescriptor(&desc)
        handle = desc!
        rankRequested = shape.rank
        let shapeComponents = shape.dimensions.map{Int32($0)}
        let strides: [Int32] = Array(repeating: 1, count: shape.rank)
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
public struct Tensor<Element : TensorDataProtocol> {

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
