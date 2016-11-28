//
//  Tensor.swift
//  DLVM
//
//  Created by Richard Wei on 9/28/16.
//
//

import CUDARuntime
import CCuDNN
import Warp

/// cuDNN tensor descriptor
/// - parameter Element: type of elements of the tensor (Float, Double, ...)
final class TensorDescriptor<Element : TensorDataProtocol> {

    let handle: cudnnTensorDescriptor_t

    /// For now, we assume that strides are always 1s.
    let strides: [Int32]

    private let originalRank: Int
    private let rankRequested: Int32

    /// Initialize a cuDNN tensor descriptor
    ///
    /// - Parameter shape: tensor shape
    init(shape: TensorShape) {
        originalRank = shape.rank
        rankRequested = Int32(Swift.max(shape.rank, 4))
        /// Strides array should have at least 4 elements
        strides = Array(repeating: 1, count: Int(rankRequested))
        var desc: cudnnTensorDescriptor_t?
        !!cudnnCreateTensorDescriptor(&desc)
        handle = desc!
        /// Components array should have at least 4 elements
        var shapeComponents: [Int32] = shape.dimensions.map{Int32($0)}
        if shapeComponents.count < 4 {
            shapeComponents.insert(
                contentsOf: repeatElement(1, count: 4 - shapeComponents.count),
                at: 0
            )
        }
        shapeComponents.withUnsafeBufferPointer { componentsBuf in
            strides.withUnsafeBufferPointer { stridesBuf in
                !!cudnnSetTensorNdDescriptor(
                    handle,
                    Element.tensorDataType.cType,
                    rankRequested,
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
        var dimensions = Array<Int32>(repeating: 1, count: Int(rankRequested))
        var strides = dimensions
        !!cudnnGetTensorNdDescriptor(
            handle,
            Int32(rankRequested),
            &dataType,
            &rank,
            &dimensions,
            &strides
        )
        return TensorShape(dimensions.map{Int($0)}.suffix(originalRank))
    }

}

public enum TensorInitializer<DataType : TensorDataProtocol> {
    case random(from: DataType, to: DataType)
    case value(DataType)
    case zeros
}

/// Tensor
/// - parameter Element: type of elements of the tensor (Float, Double, ...)
public struct DeviceTensor<Element : TensorDataProtocol> {

    /// cuDNN descriptor
    let descriptor: TensorDescriptor<Element>

    /// Tensor shape
    public let shape: TensorShape
    
    /// Contiguous storage (an array) on device
    public internal(set) var elements: DeviceArray<Element>

    /// The GPU device that owns this tensor
    public var device: Device {
        return elements.device
    }

    /// Initialize from an array on GPU device.
    /// - parameter shape: shape of the tensor
    /// - parameter storage: array on device
    public init(shape: TensorShape, storage: DeviceArray<Element>) {
        self.descriptor = TensorDescriptor(shape: shape)
        self.elements = storage
        self.shape = shape
    }

    /// Allocate and initialize a tensor to a repeated value.
    /// - parameter shape: tensor shape
    /// - parameter repeating: repeated value
    public init(shape: TensorShape,
                repeating repeatedValue: Element,
                device: Device = Device.current) {
        descriptor = TensorDescriptor(shape: shape)
        elements = DeviceArray(repeating: repeatedValue,
                               count: shape.contiguousSize,
                               device: device)
        self.shape = shape
    }
    
    /// Allocate and initialize a tensor using the factory function
    /// - parameter shape: tensor shape
    /// - parameter repeating: repeated value
    public init(shape: TensorShape,
                device: Device = Device.current,
                factory supplier: () -> Element) {
        descriptor = TensorDescriptor(shape: shape)
        let contiguousSize = shape.contiguousSize
        let hostData = (0..<contiguousSize).map { _ in supplier() }
        elements = DeviceArray(hostData, device: device)
        self.shape = shape
    }
    
    /// Allocate and initialize a tensor.
    /// - parameter shape: tensor shape
    public init(shape: TensorShape, device: Device = Device.current) {
        descriptor = TensorDescriptor(shape: shape)
        elements = DeviceArray(device: device, capacity: shape.contiguousSize)
        self.shape = shape
    }
    
    /// Compute the contiguous storage index from high-dimensional tensor indices.
    /// - parameter indices: tensor indices
    /// - returns: index in contiguous storage
    /// - note: the count of indices must equal the rank of the tensor
    private func contiguousIndex(from indices: [Int]) -> Int {
        /// Row-major order addressing
        return Int(indices.enumerated().reduce(0, { acc, next in
            next.element * (next.offset..<shape.rank).reduce(0, { $0 + shape[$1] })
        }))
    }
    
    /// Access an element of the tensor.
    /// - parameter indices: tensor indices
    /// - returns: reference to the value on GPU device
    /// - note: the count of indices must equal the rank of the tensor
    public subscript(indices: Int...) -> DeviceValue<Element> {
        get {
            precondition(indices.count == shape.rank, "Incorrect index dimension")
            return elements[contiguousIndex(from: indices)]
        }
        set {
            precondition(indices.count == shape.rank, "Incorrect index dimension")
            elements[contiguousIndex(from: indices)] = newValue
        }
    }
    
}

extension DeviceTensor {
    
    mutating func withUnsafeMutableDeviceAddress<Result>
        (_ body: (UnsafeMutablePointer<Element>) throws -> Result) rethrows -> Result {
        return try elements.withUnsafeMutableDevicePointer { ptr in
            try body(ptr.deviceAddress)
        }
    }
    
    func withUnsafeDeviceAddress<Result>
        (_ body: (UnsafePointer<Element>) throws -> Result) rethrows -> Result {
        return try elements.withUnsafeDevicePointer { ptr in
            try body(ptr.deviceAddress)
        }
    }
    
    public mutating func withUnsafeMutableDevicePointer<Result>
        (_ body: (UnsafeMutableDevicePointer<Element>) throws -> Result) rethrows -> Result {
        return try elements.withUnsafeMutableDevicePointer { ptr in
            try body(ptr)
        }
    }
    
    public func withUnsafeDevicePointer<Result>
        (_ body: (UnsafeDevicePointer<Element>) throws -> Result) rethrows -> Result {
        return try elements.withUnsafeDevicePointer { ptr in
            try body(ptr)
        }
    }
    
}

public typealias Tensor<Element : TensorDataProtocol> = DeviceTensor<Element>
