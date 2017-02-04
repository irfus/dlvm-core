//
//  Shape.swift //  DLVM
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

    public func multiplied(by other: TensorShape) -> TensorShape? {
        guard last == other.first else { return nil }
        let newDim = dimensions.dropLast() + other.dimensions.dropFirst()
        return TensorShape(newDim)
    }

    public static func ⊗ (lhs: TensorShape, rhs: TensorShape) -> TensorShape? {
        return lhs.multiplied(by: rhs)
    }

    public func matrixMultiplied(by other: TensorShape) -> TensorShape? {
        /// Has to be a matrix at least
        guard rank >= 2, other.rank >= 2 else { return nil }
        /// Match inner dimensions for matrix multiplication
        guard dropFirst().first == other.first else { return nil }
        /// Multiply inner dimensions
        var newShape = self
        newShape[1] = other[1]
        return newShape
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
