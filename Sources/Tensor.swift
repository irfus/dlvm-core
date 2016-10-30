//
//  Tensor.swift
//  Tensor
//
//  Created by Richard Wei on 9/28/16.
//
//

import CCuDNN
import CUDARuntime

public struct Shape : ExpressibleByArrayLiteral {

    public var dimensions: [Int]
    public var strides: [Int]

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

}

public class TensorDescriptor<Element : TensorDataProtocol> {

    let handle: cudnnTensorDescriptor_t

    var rankRequested: Int

    public init(shape: Shape) {
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

    public var shape: Shape {
        var rank: Int32 = 0
        var dataType = cudnnDataType_t(0)
        var dimensions = Array<Int32>(repeating: 1, count: rankRequested)
        var strides = dimensions
        cudnnGetTensorNdDescriptor(
            handle,
            Int32(rankRequested),
            &dataType,
            &rank,
            &dimensions,
            &strides
        )
        return Shape(dimensions: dimensions.map{Int($0)}, strides: strides.map{Int($0)})
    }

}
