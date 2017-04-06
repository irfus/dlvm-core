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
public protocol IRCollection : class, IRUnit, RandomAccessCollection, HashableByReference {
    associatedtype ElementCollection : RandomAccessCollection
    var elements: ElementCollection { get set }
    func append(_ element: Element)
    func insert(_ element: Element, at index: ElementCollection.Index)
    func remove(_ element: Element)
}

// MARK: - Mutation
public extension IRCollection
    where ElementCollection : OrderedMapSetProtocol,
          ElementCollection.Element == ElementCollection.Iterator.Element {

    public func append(_ element: Element) {
        elements.append(element)
    }

    public func insert(_ element: Element, at index: ElementCollection.Index) {
        elements.insert(element, at: index)
    }

    public func insert(_ element: Element, after other: Element) {
        elements.insert(element, after: other)
    }

    public func insert(_ element: Element, before other: Element) {
        elements.insert(element, before: other)
    }
    
    public func index(of element: Element) -> Int? {
        return elements.index(of: element)
    }

    public func remove(_ element: Element) {
        elements.remove(element)
    }

    public func contains(_ element: Element) -> Bool {
        return elements.contains(element)
    }

    public func element(named name: String) -> Element? {
        return elements.element(named: name)
    }

    public func containsElement(named name: String) -> Bool {
        return elements.containsElement(named: name)
    }

}

// MARK: - Mutation
public extension IRCollection
    where ElementCollection.Iterator.Element : IRSubUnit,
          ElementCollection.Iterator.Element.Parent == Self,
          ElementCollection : OrderedMapSetProtocol,
          ElementCollection.Element == ElementCollection.Iterator.Element {

    public func append(_ element: Element) {
        elements.append(element)
        element.parent = self
        invalidateAnalyses()
    }

    public func insert(_ element: Element, at index: ElementCollection.Index) {
        elements.insert(element, at: index)
        element.parent = self
        invalidateAnalyses()
    }

    public func insert(_ element: Element, after other: Element) {
        elements.insert(element, after: other)
        element.parent = self
        invalidateAnalyses()
    }

    public func insert(_ element: Element, before other: Element) {
        elements.insert(element, before: other)
        element.parent = self
        invalidateAnalyses()
    }

    public func remove(_ element: Element) {
        elements.remove(element)
        invalidateAnalyses()
    }

}

// MARK: - RandomAccessCollection default implementation
public extension IRCollection {

    public typealias Element = ElementCollection.Iterator.Element
//    public typealias Index = ElementCollection.Index // SILGen crasher
    public typealias Indices = DefaultRandomAccessIndices<ElementCollection>
    public typealias SubSequence = ElementCollection.SubSequence

    public func makeIterator() -> ElementCollection.Iterator {
        return elements.makeIterator()
    }

    public func index(after i: ElementCollection.Index) -> ElementCollection.Index {
        return elements.index(after: i)
    }

    public var indices: DefaultRandomAccessIndices<ElementCollection> {
        return elements.indices
    }

    public var startIndex: ElementCollection.Index {
        return elements.startIndex
    }

    public var endIndex: ElementCollection.Index {
        return elements.endIndex
    }

    public subscript(i: ElementCollection.Index) -> Element {
        return elements[i]
    }

    public func index(before i: ElementCollection.Index) -> ElementCollection.Index {
        return elements.index(before: i)
    }

}
