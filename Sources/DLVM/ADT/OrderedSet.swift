//
//  OrderedSet.swift
//  DLVM
//
//  Copyright 2016-2017 The DLVM Team.
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

import Foundation

public protocol OrderedSetCollection : RangeReplaceableCollection, RandomAccessCollection, MutableCollection {
    mutating func remove(_ element: Element)
    mutating func insert(_ element: Element, after other: Element)
    mutating func insert(_ element: Element, before other: Element)
}

public struct OrderedSet<Element : Hashable> : OrderedSetCollection {
    fileprivate var elements = NSMutableOrderedSet()
    fileprivate var mutatingElements: NSMutableOrderedSet {
        mutating get {
            if !isKnownUniquelyReferenced(&elements) {
                elements = elements.mutableCopy() as! NSMutableOrderedSet
            }
            return elements
        }
    }
    public init() {}

    public init<S : Sequence>(_ elements: S) where S.Iterator.Element == Element {
        append(contentsOf: elements)
    }
}

extension OrderedSet : ExpressibleByArrayLiteral {
    public init(arrayLiteral elements: Element...) {
        self.init(elements)
    }
}

extension OrderedSet : Equatable {
    public static func == (lhs: OrderedSet, rhs: OrderedSet) -> Bool {
        return lhs.elements.isEqual(to: rhs.elements)
    }
}

public extension OrderedSet {
    mutating func append(_ element: Element) {
        mutatingElements.add(element)
    }

    mutating func append<S: Sequence>(contentsOf elements: S) where S.Iterator.Element == Element {
        for element in elements {
            append(element)
        }
    }

    mutating func insert(_ element: Element, at index: Int) {
        mutatingElements.insert(element, at: index)
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
    }
    
    @discardableResult
    public mutating func remove(at position: Int) -> Element {
        defer {
            mutatingElements.removeObject(at: position)
        }
        return self[position]
    }

    mutating func removeAll() {
        mutatingElements.removeAllObjects()
    }
}

public extension OrderedSet {
    func contains(_ element: Element) -> Bool {
        return elements.contains(element)
    }

    func index(of element: Element) -> Int? {
        return elements.index(of: element)
    }

    var array: [Element] {
        return elements.array as! [Element]
    }
}

extension OrderedSet : Sequence {
    public func makeIterator() -> IndexingIterator<[Element]> {
        return array.makeIterator()
    }
}

extension OrderedSet : RandomAccessCollection {
    public typealias SubSequence = Slice<OrderedSet>

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

extension OrderedSet : MutableCollection {
    /// - Note: This is an in-house impleentation of `swapAt(_:_:)` to work around
    /// NSMutableOrderedSet's behavior that assignments of an in-set object to an index.
    /// - TODO: Reimplement OrderedSet in pure Swift.
    /// do not change the ordering.
    public mutating func swapAt(_ i: Int, _ j: Int) {
        guard i != j else { return }
        let a = self[i]
        remove(at: i)
        insert(a, at: j)
        let b = self[j-1]
        remove(at: j-1)
        insert(b, at: i)
    }
}

extension OrderedSet : RangeReplaceableCollection {
    public mutating func replaceSubrange<C>(_ subrange: Range<Int>, with newElements: C)
        where C : Collection, C.Iterator.Element == Element
    {
        var newElements = Array(newElements) as [AnyObject]
        mutatingElements.replaceObjects(in: NSRange(subrange), with: &newElements, count: subrange.count)
    }

    public subscript(bounds: Range<Int>) -> Slice<OrderedSet> {
        get {
            return Slice(base: self, bounds: bounds)
        }
        set {
            replaceSubrange(bounds, with: newValue)
        }
    }

    public subscript(position: Int) -> Element {
        get {
            return elements[position] as! Element
        }
        set {
            mutatingElements.replaceObject(at: position, with: newValue)
        }
    }
}
