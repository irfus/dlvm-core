//
//  Collection.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

public protocol EquatableByReference : class, Equatable {}
public protocol HashableByReference : EquatableByReference, Hashable {}

public extension EquatableByReference {
    public static func == (lhs: Self, rhs: Self) -> Bool {
        return lhs === rhs
    }
}

public extension HashableByReference {
    public var hashValue: Int {
        return ObjectIdentifier(self).hashValue
    }
}

public protocol IRCollection : class, RandomAccessCollection, HashableByReference {
    associatedtype Element : AnyObject
    var elements: [Element] { get }
    func append(_: Element)
    func index(of: Element) -> Int?
    func remove(_: Element)
}

public protocol IRObject : class, HashableByReference {
    associatedtype Parent : IRCollection
    weak var parent: Parent? { get }
}

// MARK: - RandomAccessCollection
public extension IRCollection {

    public typealias Index = Int

    public func index(after i: Int) -> Int {
        return elements.index(after: i)
    }

    public func index(before i: Int) -> Int {
        return elements.index(before: i)
    }

    public var indices: CountableRange<Int> {
        return elements.indices
    }

    public var startIndex: Int {
        return elements.startIndex
    }

    public var endIndex: Int {
        return elements.endIndex
    }

    public subscript(i: Int) -> Element {
        return elements[i]
    }
    
}

public extension IRObject where Parent.Element == Self {

    /// Remove self from parent basic block (if any)
    public func removeFromParent() {
        parent?.remove(self)
    }
    
}
