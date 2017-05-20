//
//  IRCollection.swift
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

/// IRCollection
public protocol IRCollection : class, IRUnit, RandomAccessCollection {
    var elements: OrderedMapSet<Element> { get set }
    func append(_ element: Element)
    func insert(_ element: Element, at index: OrderedMapSet<Element>.Index)
    func remove(_ element: Element)
}

// MARK: - Mutation
public extension IRCollection {

    func append(_ element: Element) {
        elements.append(element)
    }

    func insert(_ element: Element, at index: OrderedMapSet<Element>.Index) {
        elements.insert(element, at: index)
    }

    func insert(_ element: Element, after other: Element) {
        elements.insert(element, after: other)
    }

    func insert(_ element: Element, before other: Element) {
        elements.insert(element, before: other)
    }
    
    func index(of element: Element) -> Int? {
        return elements.index(of: element)
    }

    func remove(_ element: Element) {
        elements.remove(element)
    }

    func contains(_ element: Element) -> Bool {
        return elements.contains(element)
    }

    func element(named name: String) -> Element? {
        return elements.element(named: name)
    }

    func containsElement(named name: String) -> Bool {
        return elements.containsElement(named: name)
    }

}

public extension IRCollection {
    func makeIterator() -> OrderedMapSet<Element>.Iterator {
        return elements.makeIterator()
    }

    func index(after i: OrderedMapSet<Element>.Index) -> OrderedMapSet<Element>.Index {
        return elements.index(after: i)
    }

    var indices: DefaultRandomAccessIndices<OrderedMapSet<Element>> {
        return elements.indices
    }

    var startIndex: OrderedMapSet<Element>.Index {
        return elements.startIndex
    }

    var endIndex: OrderedMapSet<Element>.Index {
        return elements.endIndex
    }

    subscript(i: OrderedMapSet<Element>.Index) -> Element {
        return elements[i]
    }

    func index(before i: OrderedMapSet<Element>.Index) -> OrderedMapSet<Element>.Index {
        return elements.index(before: i)
    }

}


/*

// MARK: - Mutation
public extension IRCollection
    where OrderedMapSet<Element>.Element : IRSubUnit,
          OrderedMapSet<Element>.Element.Parent == Self {

    func append(_ element: Element) {
        elements.append(element)
        element.parent = self
        invalidateAnalyses()
    }

    func insert(_ element: Element, at index: OrderedMapSet<Element>.Index) {
        elements.insert(element, at: index)
        element.parent = self
        invalidateAnalyses()
    }

    func insert(_ element: Element, after other: Element) {
        elements.insert(element, after: other)
        element.parent = self
        invalidateAnalyses()
    }

    func insert(_ element: Element, before other: Element) {
        elements.insert(element, before: other)
        element.parent = self
        invalidateAnalyses()
    }

    func remove(_ element: Element) {
        elements.remove(element)
        invalidateAnalyses()
    }

    //typealias SubSequence = OrderedMapSet<Element>.SubSequence
    //typealias Index = OrderedMapSet<Element>.Index
    //typealias Indices = DefaultRandomAccessIndices<OrderedMapSet<Element>>

    func makeIterator() -> OrderedMapSet<Element>.Iterator {
        return elements.makeIterator()
    }

    func index(after i: OrderedMapSet<Element>.Index) -> OrderedMapSet<Element>.Index {
        return elements.index(after: i)
    }

    var indices: DefaultRandomAccessIndices<OrderedMapSet<Element>> {
        return elements.indices
    }

    var startIndex: OrderedMapSet<Element>.Index {
        return elements.startIndex
    }

    var endIndex: OrderedMapSet<Element>.Index {
        return elements.endIndex
    }

    subscript(i: OrderedMapSet<Element>.Index) -> Element {
        return elements[i]
    }

    func index(before i: OrderedMapSet<Element>.Index) -> OrderedMapSet<Element>.Index {
        return elements.index(before: i)
    }

}
 */
