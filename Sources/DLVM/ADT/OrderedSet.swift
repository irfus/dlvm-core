//
//  OrderedSet.swift
//  DLVM
//
//  Copyright 2016-2018 The DLVM Team.
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

public struct OrderedSet<Element : Hashable> {
    public typealias Index = Int
    public typealias Indices = CountableRange<Int>

    fileprivate var array: [Element] = []
    fileprivate var set: Set<Element> = []

    public init() {}
}

public extension OrderedSet {
    init<S : Sequence>(_ elements: S) where S.Element == Element {
        self.init()
        append(contentsOf: elements)
    }

    var count: Int {
        return array.count
    }

    var isEmpty: Bool {
        return array.isEmpty
    }

    func contains(_ member: Element) -> Bool {
        return set.contains(member)
    }

    func index(of element: Element) -> Int? {
        guard contains(element) else { return nil }
        return array.index(of: element)
    }

    @discardableResult
    mutating func append(_ element: Element) -> Bool {
        let inserted = set.insert(element).inserted
        if inserted {
            array.append(element)
        }
        return inserted
    }

    mutating func append<S: Sequence>(contentsOf elements: S)
        where S.Element == Element {
        for element in elements {
            append(element)
        }
    }

    mutating func insert(_ element: Element, at index: Int) {
        let inserted = set.insert(element).inserted
        if inserted {
            array.insert(element, at: index)
        }
    }

    mutating func insert(_ element: Element, after other: Element) {
        guard let previousIndex = index(of: other) else {
            preconditionFailure("Element to insert after is not in the set")
        }
        insert(element, at: previousIndex + 1)
    }

    mutating func insert(_ element: Element, before other: Element) {
        guard let index = index(of: other) else {
            preconditionFailure("Element to insert before is not in the set")
        }
        insert(element, at: index)
    }

    mutating func remove(_ element: Element) {
        guard let foundIndex = index(of: element) else { return }
        set.remove(element)
        array.remove(at: foundIndex)
    }

    mutating func removeAll(keepingCapacity keepCapacity: Bool) {
        array.removeAll(keepingCapacity: keepCapacity)
        set.removeAll(keepingCapacity: keepCapacity)
    }

    mutating func swapAt(_ i: Int, _ j: Int) {
        guard i != j else { return }
        let tmp = array[i]
        array[i] = array[j]
        array[j] = tmp
    }
}

extension OrderedSet : ExpressibleByArrayLiteral {
    public init(arrayLiteral elements: Element...) {
        self.init(elements)
    }
}

extension OrderedSet : Equatable {
    public static func == (lhs: OrderedSet, rhs: OrderedSet) -> Bool {
        return lhs.array == rhs.array
    }
}

extension OrderedSet : Sequence {
    public func makeIterator() -> IndexingIterator<[Element]> {
        return array.makeIterator()
    }
}

extension OrderedSet
    : RangeReplaceableCollection, BidirectionalCollection, MutableCollection {
    public func index(after i: Int) -> Int {
        return array.index(after: i)
    }

    public func index(before i: Int) -> Int {
        return array.index(before: i)
    }

    public var startIndex: Int {
        return array.startIndex
    }

    public var endIndex: Int {
        return array.endIndex
    }

    public var indices: CountableRange<Int> {
        return array.indices
    }

    public subscript(index: Int) -> Element {
        get {
            return array[index]
        }
        set {
            set.remove(array[index])
            array[index] = newValue
            set.insert(newValue)
        }
    }
}

/// Efficient initializer on Set
public extension Set {
  init(_ elements: OrderedSet<Element>) {
    self = elements.set
  }
}

/// Efficient initializer on Array
public extension Array where Element : Hashable {
  init(_ elements: OrderedSet<Element>) {
    self = elements.array
  }
}
