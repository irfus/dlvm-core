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
public struct Tensor<ItemType> {

    /// Sub-tensor (element) shape
    public internal(set) var elementShape: TensorShape {
        didSet {
            itemCountPerElement = elementShape.contiguousSize
        }
    }

    /// Tensor shape
    public var shape: TensorShape {
        return elementShape.prepending(count)
    }

    public var itemCountPerElement: Int

    /// Contiguous storage
    public internal(set) var items: ArraySlice<ItemType>

    /// Capacity reserved for sub-tensors
    public var capacity: Int {
        return items.capacity / itemCountPerElement
    }

    internal init(elementShape: TensorShape, items: ArraySlice<ItemType>) {
        let contiguousSize = elementShape.contiguousSize
        precondition(items.count % contiguousSize == 0,
                     "Item count does not match element shape")
        self.elementShape = elementShape
        self.items = items
        self.itemCountPerElement = contiguousSize
    }

    /// Initialize a tensor using an existing slice of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: slice of existing elements in row-major order
    internal init(shape: TensorShape, items: ArraySlice<ItemType>) {
        precondition(items.count >= shape.contiguousSize,
                     "The slice has fewer elements than required by the shape")
        self.init(elementShape: shape.dropFirst(), items: items.prefix(shape.contiguousSize))
    }

    /// Initialize an empty tensor of scalar elements
    public init() {
        self.init(elementShape: .scalar)
    }

    /// Initialize an empty tensor
    public init(elementShape: TensorShape) {
        self.init(elementShape: elementShape, items: [])
    }

    /// Initialize a tensor from a sequence of tensors of element shape
    public init<S : Collection>(elementShape: TensorShape, elements: S)
        where S.Iterator.Element == Tensor<ItemType> {
        self.init(elementShape: elementShape)
        self.append(contentsOf: elements)
    }

    /// Initialize a tensor from a collection of tensors of element shape
    public init<C : Collection>(elementShape: TensorShape, elements: C)
        where C.Iterator.Element == Tensor<ItemType>, C.IndexDistance == Int {
        self.init(elementShape: elementShape)
        self.append(contentsOf: elements)
    }

    /// Initialize a tensor from a sequence of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: sequence of elements in row-major order
    /// - parameter vacancySupplier
    public init<S: Sequence>(shape: TensorShape, items: S,
                             vacancySupplier supplier: (() -> ItemType)? = nil)
        where S.Iterator.Element == ItemType, S.SubSequence : Sequence,
              S.SubSequence.Iterator.Element == ItemType {
        let contiguousSize = shape.contiguousSize
        var slice = ArraySlice(items.prefix(contiguousSize))
        /// If elements fewer than required by the shape and supplier is provided
        /// generate new elements using the supplier until vacancy is filled
        if slice.count < contiguousSize, let supplier = supplier {
            slice.reserveCapacity(shape.contiguousSize)
            slice.append(contentsOf: (0..<contiguousSize).map { _ in supplier() })
        }
        self.init(shape: shape, items: slice)
    }

    /// Allocate and initialize a tensor to a repeated value
    /// - parameter shape: tensor shape
    /// - parameter repeating: repeated value
    public init(shape: TensorShape, repeating repeatedValue: ItemType) {
        let items = ArraySlice(repeating: repeatedValue, count: shape.contiguousSize)
        self.init(shape: shape, items: items)
    }

    /// Allocate and initialize a tensor using the factory function
    /// - parameter shape: tensor shape
    /// - parameter supplier: factory function providing values lazily
    public init(shape: TensorShape, supplier: () -> ItemType) {
        let items = ArraySlice((0..<shape.contiguousSize).map { _ in supplier() })
        self.init(shape: shape, items: items)
    }

    internal func itemIndex(fromIndex index: Int) -> Int {
        return itemCountPerElement * index
    }

    internal func itemSubrange(fromSubrange tensorSubrange: Range<Int>) -> Range<Int> {
        return itemIndex(fromIndex: tensorSubrange.lowerBound)
           ..< itemIndex(fromIndex: tensorSubrange.upperBound)
    }
    
}

extension Tensor where ItemType : Strideable {
    
    public init(shape: TensorShape, itemRange: Range<ItemType>) {
        var item = itemRange.lowerBound
        self.init(shape: shape, supplier: {
            defer { item = item.advanced(by: 1) }
            return item
        })
    }
    
}

// MARK: - RangeReplaceableCollection
extension Tensor : RangeReplaceableCollection {

    public var itemCount: Int {
        return items.count
    }

    public mutating func append(_ newElement: Tensor<ItemType>) {
        precondition(newElement.shape == elementShape, "Element shape mismatch")
        reserveCapacity(count + 1)
        items.append(contentsOf: newElement.items)
    }

    public mutating func reserveCapacity(_ minimumCapacity: Int) {
        let cap = Swift.max(items.capacity, itemCountPerElement * minimumCapacity)
        items.reserveCapacity(cap)
    }

    @discardableResult
    public mutating func remove(at index: Int) -> Tensor<ItemType> {
        precondition(indices.contains(index), "Index out of range")
        let range: Range = itemSubrange(fromSubrange: index..<index+1)
        defer { items.removeSubrange(range) }
        return self[range]
    }

    public mutating func append<S : Sequence>(contentsOf newElements: S)
        where S.Iterator.Element == Tensor<ItemType> {
        for element in newElements {
            append(element)
        }
    }

    public mutating func append<C : Collection>(contentsOf newElements: C)
        where C.Iterator.Element == Tensor<ItemType>, C.IndexDistance == Int {
        reserveCapacity(count + newElements.count)
        for element in newElements {
            append(element)
        }
    }

    public mutating func append(contentsOf newElements: Tensor<ItemType>) {
        precondition(newElements.elementShape == elementShape, "Element shape mismatch")
        reserveCapacity(count + newElements.count)
        items.append(contentsOf: newElements.items)
    }

    public mutating func replaceSubrange<C : Collection>
        (_ subrange: Range<Int>, with newElements: C) where C.Iterator.Element == Tensor<ItemType> {
        let storageSubrange = itemSubrange(fromSubrange: subrange)
        items.replaceSubrange(storageSubrange, with: Tensor(newElements).items)
    }

    public subscript(bounds: Range<Int>) -> Tensor<ItemType> {
        get {
            return Tensor(shape: elementShape.prepending(bounds.count),
                          items: items[itemSubrange(fromSubrange: bounds)])
        }
        set {
            precondition(newValue.shape == elementShape, "Element shape mismatch")
            items[itemSubrange(fromSubrange: bounds)] = newValue.items
        }
    }
    
}

// MARK: - RandomAccessCollection
extension Tensor : RandomAccessCollection {
    public typealias Index = Int
    public typealias SubSequence = Tensor<ItemType>

    /// Access a sub-tensor at an index specified by a list of dimensional indices
    /// - parameter indices: tensor indices
    /// - note: the count of indices must equal the rank of the tensor
    public subscript(indices: Int...) -> Tensor<ItemType> {
        get {
            return self[TensorIndex(indices)]
        }
        set {
            self[TensorIndex(indices)] = newValue
        }
    }

    /// Access a sub-tensor at index
    public subscript(index: TensorIndex) -> Tensor<ItemType> {
        get {
            let newShape = shape.dropFirst(index.count)
            let contiguousIndex = index.contiguousIndex(in: shape)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize
            return Tensor(shape: newShape, items: items[range])
        }
        set {
            let newShape = shape.dropFirst(index.count)
            let contiguousIndex = index.contiguousIndex(in: shape)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize
            items[range] = newValue.items
        }
    }
    
    /// Access a sub-tensor at the current dimension at index
    public subscript(index: Int) -> Tensor<ItemType> {
        get {
            return self[[index]]
        }
        set {
            self[[index]] = newValue
        }
    }

    public var count: Int {
        return items.count / itemCountPerElement
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

public extension Tensor {

    public func reshaped(as newShape: TensorShape) -> Tensor? {
        guard self.shape.contiguousSize == newShape.contiguousSize else {
            return nil
        }
        return Tensor(shape: newShape, items: items)
    }
    
}
