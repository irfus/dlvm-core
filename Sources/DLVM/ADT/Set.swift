//
//  Set.swift
//  DLVM
//
//  Copyright 2016-2017 Richard Wei.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

internal protocol SetImplementation {
    associatedtype Element
    associatedtype Set
    var elements: Set { get set }
}

import protocol Foundation.NSMutableCopying
import class Foundation.NSObject

internal extension SetImplementation where Set : NSObject, Set : NSMutableCopying {
    var mutatingElements: Set {
        mutating get {
            if !isKnownUniquelyReferenced(&elements) {
                elements = elements.mutableCopy() as! Set
            }
            return elements
        }
    }
}

public struct ObjectSet<Element : HashableByReference> : ExpressibleByArrayLiteral, SetImplementation {
    internal var elements: Set<Element> = []

    public init() {}

    internal init(_ elements: Set<Element>) {
        self.elements = elements
    }

    public init<S: Sequence>(_ elements: S) where S.Iterator.Element == Element {
        self.init(Set(elements))
    }

    public init(arrayLiteral elements: Element...) {
        self.init(elements)
    }

    public mutating func removeAll() {
        elements.removeAll()
    }

}

extension ObjectSet : Collection {

    public typealias Index = Set<Element>.Index

    public func index(after i: Set<Element>.Index) -> Set<Element>.Index {
        return elements.index(after: i)
    }

    public var startIndex: Set<Element>.Index {
        return elements.startIndex
    }

    public var endIndex: Set<Element>.Index {
        return elements.endIndex
    }

    public subscript(index: Index) -> Element {
        return elements[index]
    }

    public subscript(bounds: Range<Index>) -> Slice<ObjectSet<Element>> {
        return Slice(base: self, bounds: bounds)
    }

}

extension ObjectSet : Equatable {
    public static func == (lhs: ObjectSet<Element>, rhs: ObjectSet<Element>) -> Bool {
        return lhs.elements == rhs.elements
    }
}

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
        return elements.update(with: newMember)
    }

    @discardableResult
    public mutating func remove(_ member: Element) -> Element? {
        return elements.remove(member)
    }

    @discardableResult
    public mutating func insert(_ newMember: Element) -> (inserted: Bool, memberAfterInsert: Element) {
        let (inserted, member) = elements.insert(newMember)
        return (inserted, member)
    }

    public func contains(_ element: Element) -> Bool {
        return elements.contains(element)
    }

    public func contains(_ element: AnyObject) -> Bool {
        guard let element = element as? Element else { return false }
        return contains(element)
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
