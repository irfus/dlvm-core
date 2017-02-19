//
//  ObjectSet.swift
//  DLVM
//
//  Created by Richard Wei on 2/3/17.
//
//

import Foundation

fileprivate protocol ObjectSetImplementation {
    associatedtype Element
    associatedtype Set : AnyObject
    var set: Set { get set }
    var nameTable: [String : Element] { get set }
}

public protocol ObjectSetProtocol {
    associatedtype Element
    subscript(name: String) -> Element? { get }
    mutating func insert(_ element: Element)
    mutating func remove(_ element: Element)
    func contains(_ element: Element) -> Bool
    func element(named name: String) -> Element?
    @discardableResult mutating func removeElement(named name: String) -> Element?
}

public extension ObjectSetProtocol {
    public func containsValue(named name: String) -> Bool {
        return element(named: name) != nil
    }
}

fileprivate extension ObjectSetImplementation where Set : NSMutableCopying {
    var mutatingSet: Set {
        mutating get {
            if !isKnownUniquelyReferenced(&set) {
                set = set.mutableCopy() as! Set
            }
            return set
        }
    }
}

public struct NamedObjectSet<Element> : ObjectSetProtocol, ObjectSetImplementation, ExpressibleByArrayLiteral {
    fileprivate var set = NSMutableSet()
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
        return set.contains(element)
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

extension NamedObjectSet : Sequence {

    public var count: Int {
        return set.count
    }

    public func makeIterator() -> AnyIterator<Element> {
        return AnyIterator((set.lazy.map {$0 as! Element}).makeIterator())
    }

}

public struct OrderedNamedObjectSet<Element> : ObjectSetProtocol, ObjectSetImplementation, ExpressibleByArrayLiteral {
    fileprivate var set = NSMutableOrderedSet()
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

    public mutating func insert(_ element: Element) {
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
        return set.contains(element)
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
        return set.index(of: element)
    }

}

// MARK: - RandomAccessCollection, BidirectionalCollection
/// - Note: Need to conform to MutableCollection; add `subscript(i: Int)` setter
/// and `subscript(bounds: Range<Int>)`
extension OrderedNamedObjectSet : RandomAccessCollection, BidirectionalCollection {

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
        return set.count
    }

    public var indices: CountableRange<Int> {
        return 0..<set.count
    }

    public subscript(i: Int) -> Element {
        get {
            return set[i] as! Element
        }
    }

}
