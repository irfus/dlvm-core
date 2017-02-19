//
// Created by Richard Wei on 2/19/17.
//

public protocol IRCollection : class, RandomAccessCollection, HashableByReference {
    associatedtype Element
    var elements: [Element] { get }
    func append(_: Element)
    func index(of: Element) -> Int?
    func remove(_: Element)
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

public extension IRObject where Parent : IRCollection, Parent.Element == Self {

    /// Remove self from parent basic block (if any)
    public func removeFromParent() {
        parent?.remove(self)
    }

}
