//
//  Tensor.swift
//  DLVM
//
//  Created by Richard Wei on 2/3/17.
//
//

import DLVM

public struct TensorIndex : ExpressibleByArrayLiteral, Comparable {
    var elements: [Int]
    
    public init(arrayLiteral elements: Int...) {
        self.elements = elements
    }

    public init(_ indexElements: Int...) {
        self.elements = indexElements
    }

    public init<S: Sequence>(_ indexElements: S) where S.Iterator.Element == Int {
        self.elements = Array(indexElements)
    }

    public init(repeating repeatedValue: Int, count: Int) {
        self.elements = Array(repeating: repeatedValue, count: count)
    }

    /// Compute the contiguous storage index from high-dimensional tensor indices
    /// - parameter indices: tensor indices
    /// - returns: index in contiguous storage
    /// - note: the count of indices must equal the rank of the tensor
    public func contiguousIndex(in shape: TensorShape) -> Int {
        /// Row-major order addressing
        let trimmedShape = shape.prefix(count)
        return elements.enumerated().reduce(0, { acc, next -> Int in
            next.element + trimmedShape.dropFirst(next.offset).reduce(1, *)
        })
    }

    public static func ==(lhs: TensorIndex, rhs: TensorIndex) -> Bool {
        return lhs.elements == rhs.elements
    }

    public static func <(lhs: TensorIndex, rhs: TensorIndex) -> Bool {
        for (x, y) in zip(lhs.elements, rhs.elements) {
            /// Less-than at a higher dimension => true
            if x < y { return true }
            /// Greater-than at a higher dimension => false
            if x > y { return false }
            /// Otherwise, at the same higher dimension => continue
        }
        return false
    }
    
}

extension TensorIndex : RandomAccessCollection {

    public var count: Int {
        return elements.count
    }

    public subscript(bounds: Range<Int>) -> TensorIndex {
        get {
            return TensorIndex(elements[bounds])
        }
        set {
            elements[bounds] = ArraySlice(newValue.elements)
        }
    }

    public var indices: CountableRange<Int> {
        return elements.indices
    }

    public func index(after i: Int) -> Int {
        return elements.index(after: i)
    }

    public func index(before i: Int) -> Int {
        return elements.index(before: i)
    }

    public var startIndex: Int {
        return elements.startIndex
    }

    public var endIndex: Int {
        return elements.endIndex
    }

    /// Size of i-th dimension
    /// - parameter i: dimension
    public subscript(i: Int) -> Int {
        get {
            return elements[i]
        }
        set {
            elements[i] = newValue
        }
    }
    
}

public extension TensorShape {

    public var tensorIndices: AnySequence<TensorIndex> {
        return AnySequence(sequence(state: (shapeIndex: 0, indexElement: 0), next: {
            (state: inout (shapeIndex: Int, indexElement: Int)) in
            guard state.shapeIndex < self.rank else { return nil }
            if state.indexElement >= self[state.shapeIndex] {
                state.shapeIndex += 1
                state.indexElement = 0
            }
            var index = TensorIndex(self)
            index[state.shapeIndex] = state.indexElement
            state.indexElement += 1
            return index
        }))
    }
    
}

/// Tensor
public struct Tensor<Element> : RandomAccessCollection {
    public typealias Index = TensorIndex

    /// Tensor shape
    public internal(set) var shape: TensorShape
    
    /// Contiguous storage
    public internal(set) var elements: ArraySlice<Element>

    internal init(shape: TensorShape, elements: ArraySlice<Element>) {
        self.elements = elements
        self.shape = shape
    }

    public init<S: Sequence>(shape: TensorShape, elements: S)
        where S.Iterator.Element == Element {
        self.init(shape: shape, elements: ArraySlice(elements))
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
    /// - parameter repeating: repeated value
    public init(shape: TensorShape, supplier: () -> Element) {
        let contiguousSize = shape.contiguousSize
        self.elements = ArraySlice((0..<contiguousSize).map { _ in supplier() })
        self.shape = shape
    }
    

    /// Access an element of the tensor
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

    public subscript(index: TensorIndex) -> Tensor<Element> {
        get {
            let newShape = shape.dropFirst(index.count)
            let contiguousIndex = index.contiguousIndex(in: shape)
            let range = contiguousIndex..<contiguousIndex+(newShape.first ?? 1)
            return Tensor(shape: shape.dropFirst(index.count),
                          elements: elements[range])
        }
        set {
            let newShape = shape.dropFirst(index.count)
            let contiguousIndex = index.contiguousIndex(in: shape)
            let range = contiguousIndex..<contiguousIndex+(newShape.first ?? 1)
            elements[range] = newValue.elements
        }
    }

    public var indices: AnySequence<TensorIndex> {
        return shape.tensorIndices
    }

    public var startIndex: TensorIndex {
        return TensorIndex(repeating: 0, count: shape.rank)
    }

    public var endIndex: TensorIndex {
        return TensorIndex(shape)
    }

    public func index(after i: TensorIndex) -> TensorIndex {
        guard !i.isEmpty else { return i }
        var newIndex = i
        newIndex[newIndex.endIndex-1] += 1
        return newIndex
    }

    public func index(before i: TensorIndex) -> TensorIndex {
        guard !i.isEmpty else { return i }
        var newIndex = i
        newIndex[newIndex.endIndex-1] -= 1
        return newIndex
    }

}
