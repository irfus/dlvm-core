//
//  ObjectSet.swift
//  DLVM
//
//  Created by Richard Wei on 2/3/17.
//
//

import Foundation

fileprivate protocol KVSetImplementation : SetImplementation {
    var nameTable: [String : Element] { get set }
}

public protocol KVSetProtocol {
    associatedtype Element
    subscript(name: String) -> Element? { get }
    mutating func remove(_ element: Element)
    func contains(_ element: Element) -> Bool
    func element(named name: String) -> Element?
    @discardableResult mutating func removeElement(named name: String) -> Element?
}

public extension KVSetProtocol {
    public func containsValue(named name: String) -> Bool {
        return element(named: name) != nil
    }
}

public struct KVSet<Element> : KVSetProtocol, KVSetImplementation, ExpressibleByArrayLiteral {
    internal var elements = NSMutableSet()
    fileprivate var nameTable: [String : Element] = [:]

    public init() {}

    public init<S: Sequence>(_ elements: S) where S.Iterator.Element == Element {
        for element in elements {
            insert(element)
        }
    }

    public init(arrayLiteral elements: Element...) {
        self.init(elements)
    }

    public subscript(name: String) -> Element? {
        return nameTable[name]
    }

    public mutating func insert(_ element: Element) {
        mutatingSet.add(element)
        if let namedValue = element as? Named {
            nameTable[namedValue.name] = element
        }
    }
    
    public func contains(_ element: Element) -> Bool {
        return elements.contains(element)
    }

    public mutating func remove(_ element: Element) {
        mutatingSet.remove(element)
        if let namedValue = element as? Named {
            nameTable[namedValue.name] = nil
        }
    }

    public mutating func removeAll() {
        mutatingSet.removeAllObjects()
        nameTable.removeAll()
    }

    public func element(named name: String) -> Element? {
        return nameTable[name]
    }

    @discardableResult
    public mutating func removeElement(named name: String) -> Element? {
        guard let element = element(named: name) else { return nil }
        remove(element)
        return element
    }

}

extension KVSet: Sequence {

    public var count: Int {
        return elements.count
    }

    public func makeIterator() -> AnyIterator<Element> {
        return AnyIterator((elements.lazy.map {$0 as! Element}).makeIterator())
    }

}

public struct OrderedKVSet<Element> : KVSetProtocol, KVSetImplementation, ExpressibleByArrayLiteral {
    internal var elements = NSMutableOrderedSet()
    fileprivate var nameTable: [String : Element] = [:]

    public init() {}

    public init<S : Sequence>(_ elements: S) where S.Iterator.Element == Element {
        for element in elements {
            append(element)
        }
    }

    public init(arrayLiteral elements: Element...) {
        self.init(elements)
    }

    public subscript(name: String) -> Element? {
        return nameTable[name]
    }

    private mutating func insertName(of element: Element) {
        if let namedValue = element as? Named {
            nameTable[namedValue.name] = element
        }
    }

    private mutating func removeName(of element: Element) {
        if let namedValue = element as? Named {
            nameTable[namedValue.name] = nil
        }
    }

    public mutating func append(_ element: Element) {
        mutatingSet.add(element)
        insertName(of: element)
    }

    public mutating func insert(_ element: Element, after previous: Element) {
        guard let previousIndex = index(of: previous) else {
            preconditionFailure("Element to insert before is not in the set")
        }
        mutatingSet.insert(element, at: previousIndex + 1)
        insertName(of: element)
    }

    public func contains(_ element: Element) -> Bool {
        return elements.contains(element)
    }

    public mutating func remove(_ element: Element) {
        mutatingSet.remove(element)
        removeName(of: element)
    }

    public mutating func removeAll() {
        mutatingSet.removeAllObjects()
        nameTable.removeAll()
    }

    public func element(named name: String) -> Element? {
        return nameTable[name]
    }

    @discardableResult
    public mutating func removeElement(named name: String) -> Element? {
        guard let element = element(named: name) else { return nil }
        remove(element)
        return element
    }

    public func index(of element: Element) -> Int? {
        return elements.index(of: element)
    }

    public var array: [Element] {
        return elements.array as! [Element]
    }

}

// MARK: - RandomAccessCollection, BidirectionalCollection
/// - Note: Need to conform to MutableCollection; add `subscript(i: Int)` setter
/// and `subscript(bounds: Range<Int>)`
extension OrderedKVSet: RandomAccessCollection, BidirectionalCollection {

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

    public subscript(i: Int) -> Element {
        get {
            return elements[i] as! Element
        }
    }

}
