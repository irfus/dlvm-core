//
//  Tensor.swift
//  DLVM
//
//  Created by Richard Wei on 2/3/17.
//
//

import struct DLVM.TensorShape
import struct DLVM.TensorIndex

/// Tensor
public struct Tensor<Element> {
    /// Tensor shape
    public internal(set) var shape: TensorShape
    
    /// Contiguous storage
    public internal(set) var elements: ArraySlice<Element>

    /// Initialize a tensor using an existing slice of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: slice of existing elements in row-major order
    internal init(shape: TensorShape, elements: ArraySlice<Element>) {
        self.shape = shape
        precondition(elements.count >= shape.contiguousSize,
                     "The slice has fewer elements than required by the shape")
        self.elements = elements.prefix(shape.contiguousSize)
    }

    /// Initialize a tensor from a sequence of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: sequence of elements in row-major order
    /// - parameter vacancySupplier
    public init<S: Sequence>(shape: TensorShape, elements: S,
                             vacancySupplier supplier: (() -> Element)? = nil)
        where S.Iterator.Element == Element,
              S.SubSequence : Sequence,
              S.SubSequence.Iterator.Element == Element {
        var slice = ArraySlice(elements.prefix(shape.contiguousSize))
        /// If elements fewer than required by the shape and supplier is provided
        /// generate new elements using the supplier until vacancy is filled
        if slice.count < shape.contiguousSize, let supplier = supplier {
            slice.reserveCapacity(shape.contiguousSize)
            repeat {
                slice.append(supplier())
            } while slice.count < shape.contiguousSize
        }
        self.init(shape: shape, elements: slice)
    }
    
    /// Allocate and initialize a tensor to a repeated value
    /// - parameter shape: tensor shape
    /// - parameter repeating: repeated value
    public init(shape: TensorShape, repeating repeatedValue: Element) {
        self.elements = ArraySlice(repeating: repeatedValue,
                                   count: shape.contiguousSize)
        self.shape = shape
    }

    /// Allocate and initialize a tensor using the factory function
    /// - parameter shape: tensor shape
    /// - parameter supplier: factory function providing values lazily
    public init(shape: TensorShape, supplier: () -> Element) {
        let contiguousSize = shape.contiguousSize
        self.elements = ArraySlice((0..<contiguousSize).map { _ in supplier() })
        self.shape = shape
    }

}

// MARK: - RandomAccessCollection
extension Tensor : RandomAccessCollection {
    public typealias Index = Int

    /// Access a sub-tensor at an index specified by a list of dimensional indices
    /// - parameter indices: tensor indices
    /// - note: the count of indices must equal the rank of the tensor
    public subscript(indices: Int...) -> Tensor<Element> {
        get {
            return self[TensorIndex(indices)]
        }
        set {
            self[TensorIndex(indices)] = newValue
        }
    }

    /// Access a sub-tensor at index
    public subscript(index: TensorIndex) -> Tensor<Element> {
        get {
            let newShape = shape.dropFirst(index.count)
            let contiguousIndex = index.contiguousIndex(in: shape)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize
            return Tensor(shape: newShape, elements: elements[range])
        }
        set {
            let newShape = shape.dropFirst(index.count)
            let contiguousIndex = index.contiguousIndex(in: shape)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize
            elements[range] = newValue.elements
        }
    }

    /// Access a sub-tensor at the current dimension at index
    public subscript(index: Int) -> Tensor<Element> {
        get {
            return self[[index]]
        }
        set {
            self[[index]] = newValue
        }
    }

    public var count: Int {
        return shape.first ?? 0
    }

    /// Returns a sequence of tensor indices for scalar elements
    public var indices: CountableRange<Int> {
        return 0..<count
    }

    public var startIndex: Int {
        return 0
    }

    public var endIndex: Int {
        return count
    }

    /// Returns the index after the specified one in the current dimension
    public func index(after i: Int) -> Int {
        return i + 1
    }

    /// Returns the index before the specified one in the current dimension
    public func index(before i: Int) -> Int {
        return i - 1
    }

    /// Returns the index after the specified one in the last dimension
    public func index(after i: TensorIndex) -> TensorIndex {
        guard !i.isEmpty else { return i }
        var newIndex = i
        newIndex[newIndex.endIndex-1] += 1
        return newIndex
    }

    /// Returns the index before the specified one in the last dimension
    public func index(before i: TensorIndex) -> TensorIndex {
        guard !i.isEmpty else { return i }
        var newIndex = i
        newIndex[newIndex.endIndex-1] -= 1
        return newIndex
    }

}
