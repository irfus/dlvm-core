//
//  Tensor.swift
//  Tensor
//
//  Created by Richard Wei on 9/28/16.
//
//

import CCuDNN
import Warp

public struct TensorShape : ExpressibleByArrayLiteral {

    public var dimensions: [Int]

    var strides: [Int]

    public init(rank: Int) {
        dimensions = Array(repeating: 1, count: rank)
        strides = dimensions
    }

    public init(_ dimensions: Int...) {
        self.dimensions = dimensions
        strides = Array(repeating: 1, count: dimensions.count)
    }

    public init(arrayLiteral elements: Int...) {
        self.dimensions = elements
        strides = Array(repeating: 1, count: elements.count)
    }

    public init(dimensions: [Int], strides: [Int]) {
        self.dimensions = dimensions
        self.strides = strides
    }

    public subscript(i: Int) -> Int {
        return dimensions[i]
    }

    public var rank: Int {
        return dimensions.count
    }

    public var contiguousSize: Int {
        return dimensions.reduce(1, *)
    }

}

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
        return TensorShape(dimensions: dimensions.map{Int($0)}, strides: strides.map{Int($0)})
    }

}

public struct Tensor<Element : TensorDataProtocol> {

    let descriptor: TensorDescriptor<Element>

    public let shape: TensorShape
    
    private var storage: DeviceArray<Element>

    public init(shape: TensorShape, storage: DeviceArray<Element>) {
        self.descriptor = TensorDescriptor(shape: shape)
        self.storage = storage
        self.shape = shape
    }

    public init(shape: TensorShape, repeating repeatedValue: Element) {
        descriptor = TensorDescriptor(shape: shape)
        storage = DeviceArray(repeating: repeatedValue, count: shape.contiguousSize)
        self.shape = shape
    }

    public init(shape: TensorShape) {
        descriptor = TensorDescriptor(shape: shape)
        storage = DeviceArray(capacity: shape.contiguousSize)
        self.shape = shape
    }

    public subscript(indices: Int...) -> DeviceValue<Element> {
        guard indices.count <= shape.rank else {
            fatalError("Indices out of tensor dimensions")
        }
        /// Row-major order addressing
        let index = indices.enumerated().reduce(0, { acc, next in
            next.element * (next.offset..<shape.rank).reduce(0, { $0 + shape[$1] })
        })
        return storage[index]
    }

}

