//
//  Shape.swift
//  LLNM
//
//  Created by Richard Wei on 11/13/16.
//
//

/// Tensor shape
public struct TensorShape : ExpressibleByArrayLiteral {

    public var dimensions: [Int]

    /// Initialize with rank, and set the size of each dimension to 1.
    /// - parameter rank: rank of the tensor
    public init(rank: Int) {
        dimensions = Array(repeating: 1, count: rank)
    }

    /// Initialize with sizes of dimensions. The rank of the tensor
    /// is the length of the parameter list.
    /// - parameter dimensions: sizes of dimensions
    public init(dimensions: [Int]) {
        self.dimensions = dimensions
    }

    /// Initialize with sizes of dimensions. The rank of the tensor
    /// is the length of the parameter list.
    /// - parameter dimensions: sizes of dimensions
    public init(_ dimensions: Int...) {
        self.init(dimensions: dimensions)
    }

    /// Initialize with an array literal, representing the sizes of
    /// dimensions. The rank of the tensor is the length of the parameter
    /// list.
    /// - parameter dimensions: sizes of dimensions
    public init(arrayLiteral elements: Int...) {
        self.init(dimensions: elements)
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

extension TensorShape : Equatable {
    public static func ==(lhs: TensorShape, rhs: TensorShape) -> Bool {
        return lhs.dimensions == rhs.dimensions
    }
}
