//
//  Tensor.swift
//  DLVM
//
//  Created by Richard Wei on 11/13/16.
//
//

/// Tensor shape
public struct TensorShape : ExpressibleByArrayLiteral {

    var dimensions: [Int]

    /// Initialize with rank, and set the size of each dimension to 1.
    /// - parameter rank: rank of the tensor
    public init(rank: Int) {
        dimensions = Array(repeating: 1, count: rank)
    }

    /// Initialize with sizes of dimensions. The rank of the tensor
    /// is the length of the parameter list.
    /// - parameter dimensions: sizes of dimensions
    public init<C: Collection>(_ dimensions: C)
        where C.Iterator.Element == Int {
        self.dimensions = Array(dimensions)
    }

    /// Initialize with sizes of dimensions. The rank of the tensor
    /// is the length of the parameter list.
    /// - parameter dimensions: sizes of dimensions
    public init(_ dimensions: Int...) {
        self.init(dimensions)
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

extension TensorShape : RandomAccessCollection {
    
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
    /// Concatenate two tensor shapes that have the first n-1 dimensions equal,
    /// storing the sum of the last dimensions
    /// - Precondition: The first n-1 dimensions must be the same
    /// - Parameter other: shape to concatenate with
    /// - Returns: concatenated shape
    public func concatenating(with other: TensorShape) -> TensorShape? {
        guard dimensions.dropLast() == other.dimensions.dropLast() else {
            return nil
        }
        var newShape = self
        newShape[rank-1] = self[rank-1] + other[rank-1]
        return newShape
    }

    /// Form in-place concatenation with the other tensor shape, storing the
    /// sum of the last dimensions
    ///
    /// - Precondition: The first n-1 dimensions must be the same
    /// - Parameter other: shape to concatenate with
    public mutating func formConcatenation(with other: TensorShape) {
        precondition(dimensions.dropLast() == other.dimensions.dropLast(),
                     "Shapes don't match")
        self[rank-1] += other[rank-1]
    }

    public func product(with other: TensorShape) -> TensorShape? {
        guard last == other.first else { return nil }
        let newDim = dimensions.dropLast() + other.dimensions.dropFirst()
        return TensorShape(newDim)
    }

    public static func ⊗ (lhs: TensorShape, rhs: TensorShape) -> TensorShape? {
        return lhs.product(with: rhs)
    }

}
