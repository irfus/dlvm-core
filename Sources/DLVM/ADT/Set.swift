//
//  Set.swift
//  DLVM
//
//  Created by Richard Wei on 2/21/17.
//
//

internal protocol SetImplementation {
    associatedtype Element
    associatedtype Set
    var elements: Set { get set }
}

import protocol Foundation.NSMutableCopying
import class Foundation.NSObject

internal extension SetImplementation where Set : NSObject, Set : NSMutableCopying {
    var mutatingSet: Set {
        mutating get {
            if !isKnownUniquelyReferenced(&elements) {
                elements = elements.mutableCopy() as! Set
            }
            return elements
        }
    }
}

public struct ObjectSet<Element : AnyObject> : ExpressibleByArrayLiteral, SetImplementation {
    internal var elements: Set<Box<Element>> = []

    public init() {}

    internal init(_ elements: Set<Box<Element>>) {
        self.elements = elements
    }

    public init<S: Sequence>(_ elements: S) where S.Iterator.Element == Element {
        self.init(Set(elements.lazy.map{Box($0)}))
    }

    public init(arrayLiteral elements: Element...) {
        self.init(elements)
    }

    public mutating func removeAll() {
        elements.removeAll()
    }

}


// MARK: - Collection
extension ObjectSet : Collection {

    public typealias Index = Set<Box<Element>>.Index

    public func index(after i: Set<Box<Element>>.Index) -> Set<Box<Element>>.Index {
        return elements.index(after: i)
    }

    public var startIndex: Set<Box<Element>>.Index {
        return elements.startIndex
    }

    public var endIndex: Set<Box<Element>>.Index {
        return elements.endIndex
    }

    public subscript(index: Index) -> Element {
        return elements[index].object
    }

    public subscript(bounds: Range<Index>) -> Slice<ObjectSet<Element>> {
        return Slice(base: self, bounds: bounds)
    }

}

// MARK: - Equatable
extension ObjectSet : Equatable {
    public static func == (lhs: ObjectSet<Element>, rhs: ObjectSet<Element>) -> Bool {
        return lhs.elements == rhs.elements
    }
}

// MARK: - SetAlgebra
extension ObjectSet : SetAlgebra {

    public func union(_ other: ObjectSet<Element>) -> ObjectSet<Element> {
        return ObjectSet(elements.union(other.elements))
    }

    public func intersection(_ other: ObjectSet<Element>) -> ObjectSet<Element> {
        return ObjectSet(elements.intersection(other.elements))
    }

    public func symmetricDifference(_ other: ObjectSet<Element>) -> ObjectSet<Element> {
        return ObjectSet(elements.symmetricDifference(other.elements))
    }

    @discardableResult
    public mutating func update(with newMember: Element) -> Element? {
        return elements.update(with: Box(newMember))?.object
    }

    @discardableResult
    public mutating func remove(_ member: Element) -> Element? {
        return elements.remove(Box(member))?.object
    }

    @discardableResult
    public mutating func insert(_ newMember: Element) -> (inserted: Bool, memberAfterInsert: Element) {
        let (inserted, member) = elements.insert(Box(newMember))
        return (inserted, member.object)
    }

    public func contains(_ element: Element) -> Bool {
        return elements.contains(Box(element))
    }

    public mutating func formUnion(_ other: ObjectSet<Element>) {
        elements.formUnion(other.elements)
    }

    public mutating func formIntersection(_ other: ObjectSet<Element>) {
        elements.formIntersection(other.elements)
    }

    public mutating func formSymmetricDifference(_ other: ObjectSet<Element>) {
        elements.formSymmetricDifference(other.elements)
    }

    public func subtracting(_ other: ObjectSet<Element>) -> ObjectSet<Element> {
        return ObjectSet(elements.subtracting(other.elements))
    }

    public func isSubset(of other: ObjectSet<Element>) -> Bool {
        return elements.isSubset(of: other.elements)
    }

    public func isDisjoint(with other: ObjectSet<Element>) -> Bool {
        return elements.isDisjoint(with: other.elements)
    }

    public func isSuperset(of other: ObjectSet<Element>) -> Bool {
        return elements.isSuperset(of: other.elements)
    }

    public var isEmpty: Bool {
        return elements.isEmpty
    }

    public mutating func subtract(_ other: ObjectSet<Element>) {
        elements.subtract(other.elements)
    }

}
