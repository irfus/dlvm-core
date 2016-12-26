//
//  Collection.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

public protocol IRCollection : class, RandomAccessCollection {
    associatedtype Element : AnyObject
    var elements: [Element] { get }
    func append(_: Element)
    func index(of: Element) -> Int?
    func remove(_: Element)
}

public protocol IRObject : class, Hashable {
    associatedtype Parent : IRCollection
    weak var parent: Parent? { get set }
}

// MARK: - Hashable
public extension IRCollection {

    public static func == (lhs: Self, rhs: Self) -> Bool {
        return lhs === rhs
    }

    public var hashValue: Int {
        return ObjectIdentifier(self).hashValue
    }
    
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
