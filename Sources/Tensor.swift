//
//  Tensor.swift
//  Tensor
//
//  Created by Richard Wei on 9/28/16.
//
//

import CCuDNN
import CUDARuntime

public struct Shape {

    internal private(set) var components: ContiguousArray<Int>

    public init(dimensions: Int) {
        components = ContiguousArray(repeating: 1, count: dimensions)
    }

    public init(_ sizes: Int...) {
        components = ContiguousArray(sizes)
    }

    public subscript(i: Int) -> Int {
        return components[i]
    }

    public var rank: Int {
        return components.count
    }

    public var size: Int {
        return components.reduce(1, *)
    }
    
}

public class TensorBuffer<Element> {

    let descriptor: cudnnTensorDescriptor_t

    init(shape: Shape) {
        var desc: cudnnTensorDescriptor_t?
        cudnnCreateTensorDescriptor(&desc)
        self.descriptor = desc!

        var array: DeviceArray<Float>(capacity: 10)
        array.withUnsafeMutableDevicePointer { ptr in
            ptr.deviceAddress
        }
    }

}
