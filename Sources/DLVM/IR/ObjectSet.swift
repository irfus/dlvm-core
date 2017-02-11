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
    mutating func insert(_ value: Element)
    mutating func remove(_ value: Element)
    func contains(_ value: Element) -> Bool
    func value(named name: String) -> Element?
    @discardableResult mutating func removeValue(named name: String) -> Element?
}

public extension ObjectSetProtocol {
    public func containsValue(named name: String) -> Bool {
        return value(named: name) != nil
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

    public mutating func insert(_ value: Element) {
        mutatingSet.add(value)
        if let namedValue = value as? Named {
            nameTable[namedValue.name] = value
        }
    }
    
    public func contains(_ value: Element) -> Bool {
        return set.contains(value)
    }

    public mutating func remove(_ value: Element) {
        mutatingSet.remove(value)
        if let namedValue = value as? Named {
            nameTable[namedValue.name] = nil
        }
    }

    public mutating func removeAll() {
        mutatingSet.removeAllObjects()
        nameTable.removeAll()
    }

    public func value(named name: String) -> Element? {
        return nameTable[name]
    }

    @discardableResult
    public mutating func removeValue(named name: String) -> Element? {
        guard let value = value(named: name) else { return nil }
        remove(value)
        return value
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

public struct OrderedNamedObjectSet<Element> : ObjectSetProtocol, ObjectSetImplementation {
    fileprivate var set = NSMutableOrderedSet()
    fileprivate var nameTable: [String : Element] = [:]

    public init() {}

    public subscript(name: String) -> Element? {
        return nameTable[name]
    }

    public mutating func insert(_ value: Element) {
        mutatingSet.add(value)
        if let namedValue = value as? Named {
            nameTable[namedValue.name] = value
        }
    }

    public func contains(_ value: Element) -> Bool {
        return set.contains(value)
    }

    public mutating func remove(_ value: Element) {
        mutatingSet.remove(value)
        if let namedValue = value as? Named {
            nameTable[namedValue.name] = nil
        }
    }

    public mutating func removeAll() {
        mutatingSet.removeAllObjects()
        nameTable.removeAll()
    }

    public func value(named name: String) -> Element? {
        return nameTable[name]
    }

    @discardableResult
    public mutating func removeValue(named name: String) -> Element? {
        guard let value = value(named: name) else { return nil }
        remove(value)
        return value
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
