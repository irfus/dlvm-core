//
//  ObjectSet.swift
//  DLVM
//
//  Created by Richard Wei on 2/3/17.
//
//

import Foundation

fileprivate protocol MapSetImplementation: SetImplementation {
    var nameTable: [String : Element] { get set }
}

public protocol MapSetProtocol {
    associatedtype Element
    associatedtype Index
    subscript(name: String) -> Element? { get }
    mutating func remove(_ element: Element)
    func contains(_ element: Element) -> Bool
    func element(named name: String) -> Element?
    @discardableResult mutating func removeElement(named name: String) -> Element?
}

public protocol OrderedMapSetProtocol: MapSetProtocol {
    mutating func append(_ element: Element)
    mutating func append<S: Sequence>(contentsOf elements: S) where S.Iterator.Element == Element
    mutating func insert(_ element: Element, at index: Index)
    mutating func insert(_ element: Element, after other: Element)
    mutating func insert(_ element: Element, before other: Element)
    func index(of element: Element) -> Int?
}

public extension MapSetProtocol {
    public func containsElement(named name: String) -> Bool {
        return element(named: name) != nil
    }
}

public struct OrderedMapSet<Element> : OrderedMapSetProtocol, MapSetImplementation, ExpressibleByArrayLiteral {
    internal var elements = NSMutableOrderedSet()
    fileprivate var nameTable: [String : Element] = [:]
    public init() {}

    public init<S : Sequence>(_ elements: S) where S.Iterator.Element == Element {
        append(contentsOf: elements)
    }

    public init(arrayLiteral elements: Element...) {
        self.init(elements)
    }
}

// MARK: - Name map management
fileprivate extension OrderedMapSet {

    func name(of element: Element) -> String? {
        return (element as? Named)?.name ?? (element as? MaybeNamed)?.name
    }

    mutating func insertName(of element: Element) {
        if let name = name(of: element) {
            nameTable[name] = element
        }
    }

    mutating func removeName(of element: Element) {
        if let name = name(of: element) {
            nameTable[name] = nil
        }
    }

    mutating func removeDuplicate(of element: Element) {
        if let name = name(of: element), let dup = self.element(named: name) {
            remove(dup)
        }
    }

}


// MARK: - Mutation
public extension OrderedMapSet {

    mutating func append(_ element: Element) {
        removeDuplicate(of: element)
        mutatingElements.add(element)
        insertName(of: element)
    }

    mutating func append<S: Sequence>(contentsOf elements: S) where S.Iterator.Element == Element {
        for element in elements {
            append(element)
        }
    }

    mutating func insert(_ element: Element, at index: Int) {
        removeDuplicate(of: element)
        mutatingElements.insert(element, at: index)
        insertName(of: element)
    }

    mutating func insert(_ element: Element, after other: Element) {
        guard let previousIndex = index(of: other) else {
            preconditionFailure("Element to insert after is not in the set")
        }
        insert(element, at: previousIndex + 1)
    }

    mutating func insert(_ element: Element, before other: Element) {
        guard let nextIndex = index(of: other) else {
            preconditionFailure("Element to insert before is not in the set")
        }
        insert(element, at: nextIndex - 1)
    }

    mutating func remove(_ element: Element) {
        mutatingElements.remove(element)
        removeName(of: element)
    }

    mutating func removeAll() {
        mutatingElements.removeAllObjects()
        nameTable.removeAll()
    }

    @discardableResult
    mutating func removeElement(named name: String) -> Element? {
        guard let element = element(named: name) else { return nil }
        remove(element)
        return element
    }

}


public extension OrderedMapSet {

    func contains(_ element: Element) -> Bool {
        return elements.contains(element)
    }

    func element(named name: String) -> Element? {
        return nameTable[name]
    }

    subscript(name: String) -> Element? {
        return element(named: name)
    }

    func index(of element: Element) -> Int? {
        return elements.index(of: element)
    }

    var array: [Element] {
        return elements.array as! [Element]
    }

}

// MARK: - RandomAccessCollection
extension OrderedMapSet : RandomAccessCollection {

    public func index(after i: Int) -> Int {
        return i + 1
    }

    public func index(before i: Int) -> Int {
        return i - 1
    }

    public var startIndex: Int {
        return 0
    }

    public var endIndex: Int {
        return elements.count
    }

    public var indices: CountableRange<Int> {
        return 0..<elements.count
    }

}

// MARK: - MutableCollection
extension OrderedMapSet : MutableCollection {
    public subscript(position: Int) -> Element {
        get {
            return elements[position] as! Element
        }
        set {
            elements.replaceObject(at: position, with: newValue)
        }
    }
}
