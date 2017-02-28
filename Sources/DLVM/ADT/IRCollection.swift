//
// Created by Richard Wei on 2/19/17.
//

public protocol IRCollection : class, RandomAccessCollection, HashableByReference {
    associatedtype ElementCollection : RandomAccessCollection
    var elements: ElementCollection { get set }
}

// MARK: - Mutation
public extension IRCollection
    where ElementCollection : OrderedMapSetProtocol,
          ElementCollection.Element == ElementCollection.Iterator.Element {

    public func append(_ element: Element) {
        elements.append(element)
    }

    public func insert(_ element: Element, at index: ElementCollection.Index) {
        elements.insert(element, at: index)
    }

    public func insert(_ element: Element, after other: Element) {
        return elements.insert(element, after: other)
    }

    public func insert(_ element: Element, before other: Element) {
        return elements.insert(element, before: other)
    }
    
    public func index(of element: Element) -> Int? {
        return elements.index(of: element)
    }

    public func remove(_ element: Element) {
        elements.remove(element)
    }

    public func contains(_ element: Element) -> Bool {
        return elements.contains(element)
    }

    public func element(named name: String) -> Element? {
        return elements.element(named: name)
    }

    public func containsElement(named name: String) -> Bool {
        return elements.containsElement(named: name)
    }

}

// MARK: - Mutation
public extension IRCollection
    where ElementCollection.Iterator.Element : IRUnit,
          ElementCollection.Iterator.Element.Parent == Self,
          ElementCollection : OrderedMapSetProtocol,
          ElementCollection.Element == ElementCollection.Iterator.Element {

    public func append(_ element: Element) {
        elements.append(element)
        element.parent = self
    }

    public func insert(_ element: Element, at index: ElementCollection.Index) {
        defer { element.parent = self }
        elements.insert(element, at: index)
    }

    public func insert(_ element: Element, after other: Element) {
        defer { element.parent = self }
        return elements.insert(element, after: other)
    }

    public func insert(_ element: Element, before other: Element) {
        defer { element.parent = self }
        return elements.insert(element, before: other)
    }

    public func remove(_ element: Element) {
        elements.remove(element)
        element.parent = nil
    }

}

// MARK: - RandomAccessCollection default implementation
public extension IRCollection {

    public typealias Element = ElementCollection.Iterator.Element
//    public typealias Index = ElementCollection.Index // SILGen crasher
    public typealias Indices = DefaultRandomAccessIndices<ElementCollection>
    public typealias SubSequence = ElementCollection.SubSequence

    public func makeIterator() -> ElementCollection.Iterator {
        return elements.makeIterator()
    }

    public func index(after i: ElementCollection.Index) -> ElementCollection.Index {
        return elements.index(after: i)
    }

    public var indices: DefaultRandomAccessIndices<ElementCollection> {
        return elements.indices
    }

    public var startIndex: ElementCollection.Index {
        return elements.startIndex
    }

    public var endIndex: ElementCollection.Index {
        return elements.startIndex
    }

    public subscript(i: ElementCollection.Index) -> Element {
        return elements[i]
    }

    public func index(before i: ElementCollection.Index) -> ElementCollection.Index {
        return elements.index(before: i)
    }

}
