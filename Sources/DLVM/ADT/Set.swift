//
//  Set.swift
//  DLVM
//
//  Created by Richard Wei on 2/21/17.
//
//

import Foundation

internal protocol SetImplementation {
    associatedtype Element
    associatedtype Set
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

public struct ObjectSet<Element : AnyObject> : ExpressibleByArrayLiteral, SetImplementation {
    internal var set: Set<Box<Element>> = []

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
        set.insert(Box(element))
    }

    public func contains(_ element: Element) -> Bool {
        return set.contains(Box(element))
    }

    public mutating func remove(_ element: Element) {
        set.remove(Box(element))
    }

    public mutating func removeAll() {
        set.removeAll()
    }

    public var array: [Element] {
        return set.map{$0 as! Element}
    }

}


// MARK: - Collection
extension ObjectSet : Collection {

    public typealias Index = Set<Box<Element>>.Index

    public func index(after i: Set<Box<Element>>.Index) -> Set<Box<Element>>.Index {
        return set.index(after: i)
    }

    public var startIndex: Set<Box<Element>>.Index {
        return set.startIndex
    }

    public var endIndex: Set<Box<Element>>.Index {
        return set.endIndex
    }

    public subscript(index: Index) -> Element {
        return set[index].object
    }

    public subscript(bounds: Range<Index>) -> Slice<ObjectSet<Element>> {
        return Slice(base: self, bounds: bounds)
    }

}
