//
//  Set.swift
//  DLVM
//
//  Created by Richard Wei on 2/21/17.
//
//

public typealias HashSet = Swift.Set

import Foundation

internal protocol SetImplementation {
    associatedtype Element
    associatedtype Set : AnyObject
    var set: Set { get set }
}

internal extension SetImplementation where Set : NSMutableCopying {
    var mutatingSet: Set {
        mutating get {
            if !isKnownUniquelyReferenced(&set) {
                set = set.mutableCopy() as! Set
            }
            return set
        }
    }
}

public struct Set<Element> : ExpressibleByArrayLiteral, SetImplementation {
    internal var set = NSMutableSet()

    public init() {}

    public init<S: Sequence>(_ elements: S) where S.Iterator.Element == Element {
        for element in elements {
            insert(element)
        }
    }

    public init(arrayLiteral elements: Element...) {
        self.init(elements)
    }

    public mutating func insert(_ element: Element) {
        mutatingSet.add(element)
    }

    public func contains(_ element: Element) -> Bool {
        return set.contains(element)
    }

    public mutating func remove(_ element: Element) {
        mutatingSet.remove(element)
    }

    public mutating func removeAll() {
        mutatingSet.removeAllObjects()
    }

    public var array: [Element] {
        return set.map{$0 as! Element}
    }

}


// MARK: - MutableCollection
extension Set: Sequence {

    public func makeIterator() -> AnyIterator<Element> {
        let iterator = set.makeIterator()
        return AnyIterator {
            guard let next = iterator.next() else { return nil }
            return (next as! Element)
        }
    }
    
}
