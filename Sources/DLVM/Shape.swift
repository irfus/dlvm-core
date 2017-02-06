//
//  Shape.swift
//  DLVM
//
//  Created by Richard Wei on 11/13/16.
//
//

/// Tensor shape
public struct TensorShape : ExpressibleByArrayLiteral {

    public typealias SubSequence = TensorShape

    var dimensions: [Int]

    /// Initialize with rank, and set the size of each dimension to 1.
    /// - parameter rank: rank of the tensor
    fileprivate init(rank: Int) {
        dimensions = Array(repeating: 1, count: rank)
    }

    /// Initialize with sizes of dimensions. The rank of the tensor
    /// is the length of the parameter list.
    /// - parameter dimensions: sizes of dimensions
    public init<C: Collection>(_ dimensions: C)
        where C.Iterator.Element == Int {
        self.dimensions = Array(dimensions)
    }

    /// Initialize with an array literal, representing the sizes of
    /// dimensions. The rank of the tensor is the length of the parameter
    /// list.
    /// - parameter dimensions: sizes of dimensions
    public init(arrayLiteral elements: Int...) {
        self.init(elements)
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

extension TensorShape {

    public static let scalar: TensorShape = []

    public static func vector(ofSize size: Int) -> TensorShape {
        return [size]
    }

    public static func matrix(rowCount: Int, columnCount: Int) -> TensorShape {
        return [rowCount, columnCount]
    }

    public var isScalar: Bool {
        return rank == 0
    }

    public var isVector: Bool {
        return rank == 1
    }

    public var isMatrix: Bool {
        return rank == 2
    }

}

extension TensorShape : RandomAccessCollection {

    public subscript(bounds: Range<Int>) -> TensorShape {
        get {
            return TensorShape(dimensions[bounds])
        }
        set {
            dimensions[bounds] = ArraySlice(newValue.dimensions)
        }
    }

    public var indices: CountableRange<Int> {
        return dimensions.indices
    }

    public func index(after i: Int) -> Int {
        return dimensions.index(after: i)
    }

    public func index(before i: Int) -> Int {
        return dimensions.index(before: i)
    }

    public var startIndex: Int {
        return dimensions.startIndex
    }

    public var endIndex: Int {
        return dimensions.endIndex
    }

    /// Size of i-th dimension
    /// - parameter i: dimension
    public subscript(i: Int) -> Int {
        get {
            return dimensions[i]
        }
        set {
            dimensions[i] = newValue
        }
    }

}

extension TensorShape : Equatable {

    public static func ==(lhs: TensorShape, rhs: TensorShape) -> Bool {
        return lhs.dimensions == rhs.dimensions
    }
}

infix operator ⊗ : MultiplicationPrecedence

public extension TensorShape {

    public func droppingDimension(_ dimension: Int) -> TensorShape {
        precondition(indices.contains(dimension), "Dimension index out of range")
        var newDims = dimensions
        newDims.remove(at: dimension)
        return TensorShape(newDims)
    }

    public func compatibleShape(with other: TensorShape) -> TensorShape {
        if rank >= other.rank { return self }
        var newShape = TensorShape(rank: other.rank)
        newShape[other.rank-rank..<other.rank] = other
        return newShape
    }

    /// Concatenate two tensor shapes that have every dimension equal except
    /// the specified dimension to concatenate along (the last dimension, by
    /// default)
    ///
    /// - Parameter other: shape to concatenate with
    /// - Returns: concatenated shape, or nil if dimensions don't match
    public func concatenating(with other: TensorShape,
                              alongDimension dim: Int = 0) -> TensorShape? {
        guard dimensions.prefix(dim) == other.dimensions.prefix(dim),
            dimensions.suffix(from: dim+1) == other.dimensions.suffix(from: dim+1) else {
            return nil // Dimension mismatch
        }
        var newShape = self
        newShape[dim] = self[dim] + other[dim]
        return newShape
    }

    /// Form in-place concatenation with the other tensor shape
    ///
    /// - Precondition: The dimensions except the specified axis must be equal
    /// - Parameter other: shape to concatenate with
    public mutating func formConcatenation(with other: TensorShape,
                                           alongDimension dim: Int = 0) {
        precondition(
            dimensions.prefix(dim) == other.dimensions.prefix(dim) &&
                dimensions.suffix(from: dim+1) == other.dimensions.suffix(from: dim+1),
            "Cannot concatenate due to shape mismatch"
        )
        self[dim] += other[dim]
    }

    public func canConcatenate(with other: TensorShape) -> Bool {
        return concatenating(with: other) != nil
    }

    public func multiplied(with other: TensorShape) -> TensorShape? {
        guard last == other.first else { return nil }
        let newDim = dimensions.dropLast() + other.dimensions.dropFirst()
        return TensorShape(newDim)
    }

    public static func ⊗ (lhs: TensorShape, rhs: TensorShape) -> TensorShape? {
        return lhs.multiplied(with: rhs)
    }

    public func canMultiply(with other: TensorShape) -> Bool {
        return multiplied(with: other) != nil
    }

    public func matrixMultiplied(with other: TensorShape) -> TensorShape? {
        /// Has to be a matrix at least
        guard rank >= 2, other.rank >= 2 else { return nil }
        /// Match inner dimensions for matrix multiplication
        guard dropFirst().first == other.first else { return nil }
        /// Multiply inner dimensions
        var newShape = self
        newShape[1] = other[1]
        return newShape
    }

    public func canMatrixMultiply(with other: TensorShape) -> Bool {
        return matrixMultiplied(with: other) != nil
    }

    /// Matrix/vector transpose
    public var transpose: TensorShape? {
        guard rank <= 2 else { return nil }
        if rank == 2 { return [self[1], self[0]] }
        else if rank == 1 { return [self[1], 1] }
        return self
    }

    public func canBroadcast(to other: TensorShape) -> Bool {
        return rank <= other.rank && other.suffix(rank).elementsEqual(self)
    }

    public func broadcasted(to other: TensorShape) -> TensorShape? {
        return canBroadcast(to: other) ? other : nil
    }

}

/// Tensor index
public struct TensorIndex : ExpressibleByArrayLiteral {
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
            let stride = trimmedShape.isEmpty ? 0 : trimmedShape.dropFirst(next.offset+1).reduce(1, *)
            return acc + next.element * stride
        })
    }

}

// MARK: - Equatable, Comparable
extension TensorIndex : Comparable {
    
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

// MARK: - RandomAccessCollection
extension TensorIndex : RandomAccessCollection {

    public var count: Int {
        return elements.count
    }

    public var dimension: Int {
        return count - 1
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

// MARK: - Strideable
extension TensorIndex : Strideable {

    public typealias Stride = Int

    /// Returns a `Self` `x` such that `self.distance(to: x)` approximates `n`.
    ///
    /// If `Stride` conforms to `Integer`, then `self.distance(to: x) == n`.
    ///
    /// - Complexity: O(1).
    public func advanced(by n: Int) -> TensorIndex {
        guard !isEmpty else { return self }
        var newIndex = self
        newIndex[newIndex.endIndex-1] += n
        return newIndex
    }

    /// Returns a stride `x` such that `self.advanced(by: x)` approximates
    /// `other`.
    ///
    /// If `Stride` conforms to `Integer`, then `self.advanced(by: x) == other`.
    ///
    /// - Complexity: O(1).
    public func distance(to other: TensorIndex) -> Int {
        precondition(count == other.count, "Indices are not in the same dimension")
        guard let otherLast = other.last, let selfLast = last else { return 0 }
        return otherLast - selfLast
    }

}

public extension TensorShape {

    /// Returns a sequence of tensor indices for scalar elements
    public var scalarIndices: AnySequence<TensorIndex> {
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

    /// Returns a sequence of tensor indices in the specified dimension
    public func tensorIndices(inDimension dimension: Int) -> AnySequence<TensorIndex> {
        return prefix(dimension+1).scalarIndices
    }

    /// Returns the row-major order index for the specified tensor index
    /// - parameter index: tensor index
    public func contiguousIndex(for index: TensorIndex) -> Int {
        return index.contiguousIndex(in: self)
    }
    
}
