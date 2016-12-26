//
//  Collection.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

public protocol IRCollection : class, Hashable, RandomAccessCollection, MutableCollection {
    associatedtype Element
    var elements: [Element] { get set }
}

public protocol IRObject : class, Hashable {
    associatedtype Parent : IRCollection, AnyObject
    weak var parent: Parent? { get set }
}

// MARK: - Hashable
public extension IRCollection where Self : AnyObject {

    public static func == (lhs: Self, rhs: Self) -> Bool {
        return lhs === rhs
    }

    public var hashValue: Int {
        return ObjectIdentifier(self).hashValue
    }
    
}

// MARK: - RandomAccessCollection, MutableCollection
public extension IRCollection {

    public typealias Index = Int
    public typealias SubSequence = ArraySlice<Element>
    public typealias Indices = CountableRange<Int>

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

    public var indices: CountableRange<Int> {
        return elements.indices
    }

    public subscript(bounds: Range<Int>) -> ArraySlice<Element> {
        get {
            return elements[bounds]
        }
        set {
            elements[bounds] = newValue
        }
    }

    public subscript(i: Int) -> Element {
        get {
            return elements[i]
        }
        set {
            elements[i] = newValue
        }
    }
    
}

public extension IRCollection where Element : IRObject, Element.Parent == Self {

    /// Append instruction to the end
    ///
    /// - Parameter instruction: instruction to append
    public func append(_ element: Element) {
        element.parent = self
        elements.append(element)
    }

    /// Index of the instruction
    ///
    /// - Note: Basic block is usually extremely small, we'll just do O(n)
    /// search for now
    /// - Parameter instruction: instruction to find
    /// - Returns: index of instruction
    /// - Complexity: O(n)
    public func index(of instruction: Element) -> Int? {
        return elements.index(of: instruction)
    }

    /// Remove instruction from the basic block
    ///
    /// - Precondition: instruction ∈ block
    /// - Parameter instruction: instruction to remove from the block
    public func remove(_ instruction: Element) {
        let index = self.index(of: instruction)!
        elements.remove(at: index)
    }

}

public extension IRObject where Parent.Element == Self {

    /// Remove self from parent basic block (if any)
    public func removeFromParent() {
        parent?.remove(self)
    }
    
}
